"""
法律文档智能分块器
"""
import re
from typing import List, Optional
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class LegalDocumentSplitter:
    """法律文档智能分块器"""

    # 法律条文模式
    LEGAL_PATTERNS = [
        r"第[一二三四五六七八九十百千\d]+条",  # 第X条
        r"第[一二三四五六七八九十百千\d]+章",  # 第X章
        r"第[一二三四五六七八九十百千\d]+节",  # 第X节
        r"第[一二三四五六七八九十百千\d]+款",  # 第X款
        r"Article\s+\d+",  # Article X
        r"Section\s+\d+",  # Section X
    ]

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        legal_split: bool = True
    ):
        """
        初始化分块器

        Args:
            chunk_size: 分块大小（字符数）
            chunk_overlap: 重叠大小
            legal_split: 是否启用法律文档特殊分块
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.legal_split = legal_split

        # 通用文本分块器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", "。", "；", "，", " "]
        )

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        分块文档列表

        Args:
            documents: 原始文档列表

        Returns:
            分块后的文档列表
        """
        if self.legal_split:
            return self._legal_aware_split(documents)
        else:
            return self._simple_split(documents)

    def _legal_aware_split(self, documents: List[Document]) -> List[Document]:
        """
        法律文档智能分块
        优先按法律条款分块，保持条款完整性
        """
        all_chunks = []

        for doc in documents:
            # 尝试按法律条款分块
            chunks = self._split_by_legal_articles(doc)

            if len(chunks) > 1:
                # 成功按条款分块
                all_chunks.extend(chunks)
            else:
                # 无法按条款分块，使用通用分块
                general_chunks = self.text_splitter.split_documents([doc])
                all_chunks.extend(general_chunks)

        # 添加分块元数据
        for i, chunk in enumerate(all_chunks):
            chunk.metadata["chunk_index"] = i
            chunk.metadata["chunk_size"] = len(chunk.page_content)

        return all_chunks

    def _split_by_legal_articles(self, document: Document) -> List[Document]:
        """
        按法律条款分块 — 条款完整性优先于字数限制

        核心原则：
        - 优先在"第X条/章/节/款"边界分割
        - 如果单条法条超过 chunk_size，允许其保持完整，禁止强行切断
        - 多个短条款可以合并进一个 chunk，但不能超过 chunk_size * 1.5
        """
        text = document.page_content

        # 构建条款分割正则
        combined_pattern = "|".join(self.LEGAL_PATTERNS)

        # 使用条款标题作为分割点
        parts = re.split(f"({combined_pattern})", text)

        if len(parts) <= 1:
            return [document]

        chunks = []
        current_chunk = ""
        is_legal_article = False  # 标记当前 chunk 是否以法条模式开头

        for i, part in enumerate(parts):
            if re.match(combined_pattern, part):
                # 这是一个条款标题（如"第X条"）
                if current_chunk:
                    chunks.append(self._create_chunk(document, current_chunk))
                current_chunk = part
                is_legal_article = True  # 新的法条开始
            else:
                current_chunk += part

            # ── 超长处理 ──
            if len(current_chunk) > self.chunk_size:
                if is_legal_article:
                    # 单条法条：允许保持完整，不切断
                    continue
                else:
                    # 非法条文本：允许到 1.5 倍后再保存
                    if len(current_chunk) > int(self.chunk_size * 1.5):
                        chunks.append(self._create_chunk(document, current_chunk))
                        current_chunk = ""
                        is_legal_article = False

        # 保存最后一块
        if current_chunk:
            chunks.append(self._create_chunk(document, current_chunk))

        return chunks

    def _simple_split(self, documents: List[Document]) -> List[Document]:
        """简单通用分块"""
        return self.text_splitter.split_documents(documents)

    def _create_chunk(self, original_doc: Document, content: str) -> Document:
        """创建分块文档"""
        return Document(
            page_content=content.strip(),
            metadata={
                **original_doc.metadata,
                "chunk_size": len(content),
            }
        )

    def split_text(self, text: str, metadata: Optional[dict] = None) -> List[Document]:
        """
        分块纯文本

        Args:
            text: 文本内容
            metadata: 元数据

        Returns:
            分块后的文档列表
        """
        doc = Document(page_content=text, metadata=metadata or {})
        return self.split_documents([doc])
