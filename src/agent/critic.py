"""
Critic Agent — 法条引用校验（本地核对 + 联网兜底 + 脏数据过滤）

工作流：
  1. 提取 Generator 初稿中的法条引用（《XXX法》第X条）
  2. 逐条在本地知识库核对 → 命中则精确比对
  3. 本地无此法 → 联网搜索核验 → 相关性过滤 → 标记来源
  4. 发现不一致 → 生成修正指令打回 Generator
"""
import re
from typing import List, Dict, Tuple


class CriticReview:
    """Critic 审核结果"""

    def __init__(self, passed: bool, corrections: str, refs: List[Dict]):
        self.passed = passed
        self.corrections = corrections
        self.refs = refs


class LegalCritic:
    """法条引用校验 Agent"""

    CITATION_RE = re.compile(
        r'《([^》]+)》\s*(?:第[\u4e00-\u9fff\d]+条(?:\s*之\s*[一二三\d]+)?)?'
    )

    # ── 权威法律域名（优先采信） ──
    TRUSTED_DOMAINS = [
        "court.gov.cn", "pkulaw.com", "chinalawinfo.com",
        "baike.baidu.com", "gov.cn", "pkulaw.cn",
        "npc.gov.cn", "moj.gov.cn", "chinalaw.gov.cn",
    ]

    # ── 黑名单域名（绝对剔除） ──
    BLOCKED_DOMAINS = [
        "wikipedia.org", "reddit.com", "quora.com", "youtube.com",
        "facebook.com", "twitter.com", "instagram.com", "tiktok.com",
        "amazon.com", "ebay.com", "shopify.com", "aliexpress.com",
    ]

    # ── 法律相关性关键词（网页标题/摘要必须包含至少 1 个） ──
    LEGAL_RELEVANCE_KW = [
        "法", "条", "款", "诉讼", "法院", "判决", "审理",
        "法规", "规定", "处罚", "赔偿", "责任", "犯罪",
        "合同", "劳动", "交通", "刑事", "民事", "行政",
        "司法解释", "裁定", "立法", "法律",
    ]

    def __init__(self, vector_store, llm):
        self.vector_store = vector_store
        self.llm = llm

    def review(self, draft_answer: str, query: str) -> CriticReview:
        citations = self._extract_citations(draft_answer)
        if not citations:
            return CriticReview(passed=True, corrections="", refs=[])

        corrections = []
        refs = []

        for cite_text, law_name, article_num in citations:
            local_result = self._check_local(law_name, article_num)
            if local_result["found"]:
                if not local_result["match"]:
                    corrections.append(
                        f"《{law_name}》第{article_num}条 引用有误，"
                        f"本地库原文：{local_result['actual'][:100]}"
                    )
                continue

            web_result = self._check_web(law_name, article_num, query)
            if web_result["found"]:
                refs.append({
                    "url": web_result.get("url", ""),
                    "title": web_result.get("title", ""),
                })

        correction_text = ""
        if corrections:
            correction_text = (
                "【Critic 审核】以下法条引用需要修正，请重新输出：\n"
                + "\n".join(f"- {c}" for c in corrections)
                + "\n\n请修正后重新输出完整的 Final Answer。"
            )

        return CriticReview(
            passed=len(corrections) == 0,
            corrections=correction_text,
            refs=refs,
        )

    def _extract_citations(self, text: str) -> List[Tuple[str, str, str]]:
        results = []
        seen = set()
        for m in self.CITATION_RE.finditer(text):
            raw = m.group(0)
            law_name = m.group(1).strip()
            article_m = re.search(r'第([\u4e00-\u9fff\d]+)条', raw)
            article_num = article_m.group(1) if article_m else ""
            key = f"{law_name}:{article_num}"
            if key not in seen:
                seen.add(key)
                results.append((raw, law_name, article_num))
        return results

    def _check_local(self, law_name: str, article_num: str) -> dict:
        try:
            q = f"{law_name} 第{article_num}条" if article_num else law_name
            results = self.vector_store.similarity_search_with_score(q, k=3)
            for doc, score in results:
                if law_name in doc.page_content:
                    if article_num:
                        if f"第{article_num}条" in doc.page_content:
                            return {"found": True, "match": True, "actual": doc.page_content[:200]}
                        else:
                            return {"found": True, "match": False, "actual": doc.page_content[:200]}
                    return {"found": True, "match": True, "actual": doc.page_content[:200]}
            return {"found": False, "match": None}
        except Exception as e:
            print(f"[Critic] local check error: {e}")
            return {"found": False, "match": None}

    def _check_web(self, law_name: str, article_num: str, query: str) -> dict:
        """联网核验（带脏数据过滤）"""
        try:
            from duckduckgo_search import DDGS
            search_q = f"{law_name} 第{article_num}条 原文" if article_num else f"{law_name} 全文"
            with DDGS() as ddgs:
                raw_results = list(ddgs.text(search_q, max_results=5))
            filtered = self._filter_results(raw_results, law_name)
            if filtered:
                r = filtered[0]
                return {
                    "found": True, "match": True,
                    "url": r.get("href", ""),
                    "title": r.get("title", ""),
                    "snippet": r.get("body", ""),
                }
            return {"found": False, "match": None}
        except Exception as e:
            print(f"[Critic] web check error: {e}")
            return {"found": False, "match": None}

    def _filter_results(self, results: list, law_name: str) -> list:
        """
        多层过滤：
        1. 黑名单域名 → 剔除
        2. 标题/摘要必须含中文法律关键词
        3. 标题/摘要必须含正在查找的法名中的关键字
        4. 优先采信权威法律域名
        """
        law_key = law_name.replace("中华人民共和国", "").replace("中国", "")

        def is_relevant(r: dict) -> bool:
            url = r.get("href", "")
            title = r.get("title", "")
            body = r.get("body", "")
            combined = f"{title} {body}"

            # 1. 域名黑名单
            for blocked in self.BLOCKED_DOMAINS:
                if blocked in url.lower():
                    return False

            # 2. 必须包含中文
            has_chinese = bool(re.search(r'[\u4e00-\u9fff]', combined))
            if not has_chinese:
                return False

            # 3. 必须包含法律相关性关键词
            has_legal_kw = any(kw in combined for kw in self.LEGAL_RELEVANCE_KW)
            if not has_legal_kw:
                return False

            # 4. 标题或摘要必须包含正在查找的法名关键字
            if law_key and len(law_key) > 1:
                if law_key not in combined:
                    return False

            return True

        def trust_score(r: dict) -> int:
            url = r.get("href", "")
            for trusted in self.TRUSTED_DOMAINS:
                if trusted in url:
                    return 10  # 权威来源
            if "baidu.com" in url:
                return 5   # 百科类
            return 0

        # 过滤 + 排序：可信度高的排前面
        relevant = [r for r in results if is_relevant(r)]
        relevant.sort(key=trust_score, reverse=True)
        return relevant
