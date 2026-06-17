"""
网页爬虫模块 - 用于爬取法律网站内容
"""
import re
from typing import List, Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document


class WebCrawler:
    """法律网页爬虫"""

    # 常用法律网站
    LEGAL_SITES = {
        "flk": "https://flk.npc.gov.cn/",  # 国家法律法规数据库
        "pkulaw": "https://www.pkulaw.com/",  # 北大法宝
        "wenshu": "https://wenshu.court.gov.cn/",  # 中国裁判文书网
    }

    def __init__(self, timeout: int = 30):
        """
        初始化爬虫

        Args:
            timeout: 请求超时时间（秒）
        """
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })

    def crawl_page(self, url: str, selector: Optional[str] = None) -> Optional[Document]:
        """
        爬取单个页面

        Args:
            url: 页面URL
            selector: CSS选择器，用于提取特定内容

        Returns:
            文档对象
        """
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            response.encoding = response.apparent_encoding

            soup = BeautifulSoup(response.text, "html.parser")

            # 移除脚本和样式
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()

            # 提取标题
            title = soup.find("title")
            title_text = title.get_text() if title else "无标题"

            # 提取内容
            if selector:
                content_elem = soup.select_one(selector)
                content = content_elem.get_text() if content_elem else ""
            else:
                # 尝试提取主要内容
                content_elem = soup.find("article") or soup.find("main") or soup.find("div", class_="content")
                if content_elem:
                    content = content_elem.get_text()
                else:
                    content = soup.get_text()

            # 清理文本
            content = self._clean_text(content)

            if not content:
                return None

            return Document(
                page_content=content,
                metadata={
                    "source": url,
                    "title": title_text,
                    "file_type": "webpage",
                }
            )

        except Exception as e:
            print(f"爬取失败: {url} - {str(e)}")
            return None

    def crawl_legal_article(self, url: str) -> Optional[Document]:
        """
        爬取法律条文页面（针对法律网站优化）

        Args:
            url: 法律条文URL

        Returns:
            文档对象
        """
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            response.encoding = response.apparent_encoding

            soup = BeautifulSoup(response.text, "html.parser")

            # 提取法律标题
            title = self._extract_legal_title(soup)

            # 提取法律正文
            content = self._extract_legal_content(soup)

            if not content:
                return None

            return Document(
                page_content=content,
                metadata={
                    "source": url,
                    "title": title,
                    "file_type": "legal_article",
                    "doc_type": "law",
                }
            )

        except Exception as e:
            print(f"爬取法律条文失败: {url} - {str(e)}")
            return None

    def crawl_batch(self, urls: List[str], selector: Optional[str] = None) -> List[Document]:
        """
        批量爬取页面

        Args:
            urls: URL列表
            selector: CSS选择器

        Returns:
            文档列表
        """
        documents = []
        for url in urls:
            doc = self.crawl_page(url, selector)
            if doc:
                documents.append(doc)
                print(f"✓ 已爬取: {url}")
            else:
                print(f"✗ 爬取失败: {url}")
        return documents

    def _extract_legal_title(self, soup: BeautifulSoup) -> str:
        """提取法律标题"""
        # 尝试多种选择器
        selectors = [
            "h1.title",
            "h1",
            ".law-title",
            ".title",
        ]
        for selector in selectors:
            elem = soup.select_one(selector)
            if elem:
                return elem.get_text().strip()
        return "未知法律"

    def _extract_legal_content(self, soup: BeautifulSoup) -> str:
        """提取法律正文"""
        # 尝试多种选择器
        selectors = [
            ".law-content",
            ".article-content",
            "article",
            "main",
            ".content",
        ]
        for selector in selectors:
            elem = soup.select_one(selector)
            if elem:
                return self._clean_text(elem.get_text())
        return ""

    def _clean_text(self, text: str) -> str:
        """清理文本"""
        # 移除多余空白
        text = re.sub(r"\s+", " ", text)
        # 移除空行
        text = re.sub(r"\n\s*\n", "\n", text)
        return text.strip()
