"""
文档加载器 - 支持PDF、Word、Markdown等格式
"""
import os
from pathlib import Path
from typing import List, Optional, Union
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredMarkdownLoader
)
from langchain_core.documents import Document


class DocumentLoader:
    """文档加载器"""

    def __init__(self, data_dir: str = "./data"):
        """
        初始化文档加载器

        Args:
            data_dir: 数据目录路径
        """
        self.data_dir = Path(data_dir)
        self._loaders = {
            ".pdf": self._load_pdf,
            ".docx": self._load_docx,
            ".doc": self._load_docx,
            ".md": self._load_markdown,
            ".txt": self._load_text,
        }

    def load_file(self, file_path: Union[str, Path]) -> List[Document]:
        """
        加载单个文件

        Args:
            file_path: 文件路径

        Returns:
            文档列表
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        suffix = file_path.suffix.lower()
        if suffix not in self._loaders:
            raise ValueError(f"不支持的文件格式: {suffix}")

        loader_func = self._loaders[suffix]
        return loader_func(file_path)

    def load_directory(self, dir_path: Optional[Union[str, Path]] = None) -> List[Document]:
        """
        加载目录下所有支持的文档

        Args:
            dir_path: 目录路径，默认使用初始化时的data_dir

        Returns:
            文档列表
        """
        if dir_path is None:
            dir_path = self.data_dir
        else:
            dir_path = Path(dir_path)

        if not dir_path.exists():
            raise FileNotFoundError(f"目录不存在: {dir_path}")

        all_documents = []
        supported_extensions = {".pdf", ".docx", ".doc", ".md", ".txt"}

        for file_path in dir_path.rglob("*"):
            if file_path.suffix.lower() in supported_extensions:
                try:
                    docs = self.load_file(file_path)
                    # 根据子目录添加 doc_type 元数据
                    doc_type = self._detect_doc_type(file_path, dir_path)
                    for doc in docs:
                        doc.metadata["doc_type"] = doc_type
                    all_documents.extend(docs)
                    print(f"[OK] 已加载: {file_path.name} [{doc_type}] ({len(docs)} 个文档块)")
                except Exception as e:
                    print(f"[FAIL] 加载失败: {file_path.name} - {str(e)}")

        return all_documents

    @staticmethod
    def _detect_doc_type(file_path: Path, data_dir: Path) -> str:
        """根据文件所在子目录检测文档类型"""
        try:
            relative = file_path.relative_to(data_dir)
            parts = relative.parts
            if len(parts) > 1:
                parent = parts[0].lower()
                if parent == "laws":
                    return "law"
                elif parent == "cases":
                    return "case"
                elif parent == "custom":
                    return "custom"
        except ValueError:
            pass
        return "unknown"

    def _load_pdf(self, file_path: Path) -> List[Document]:
        """加载PDF文件"""
        loader = PyPDFLoader(str(file_path))
        documents = loader.load()
        # 添加元数据
        for doc in documents:
            doc.metadata["source"] = str(file_path)
            doc.metadata["file_type"] = "pdf"
        return documents

    def _load_docx(self, file_path: Path) -> List[Document]:
        """加载Word文档"""
        loader = Docx2txtLoader(str(file_path))
        documents = loader.load()
        for doc in documents:
            doc.metadata["source"] = str(file_path)
            doc.metadata["file_type"] = "docx"
        return documents

    def _load_markdown(self, file_path: Path) -> List[Document]:
        """加载Markdown文件"""
        try:
            loader = UnstructuredMarkdownLoader(str(file_path))
            documents = loader.load()
        except (ImportError, ModuleNotFoundError):
            # 缺少 unstructured 包时回退到纯文本加载
            loader = TextLoader(str(file_path), encoding="utf-8")
            documents = loader.load()
        for doc in documents:
            doc.metadata["source"] = str(file_path)
            doc.metadata["file_type"] = "markdown"
        return documents

    def _load_text(self, file_path: Path) -> List[Document]:
        """加载纯文本文件"""
        loader = TextLoader(str(file_path), encoding="utf-8")
        documents = loader.load()
        for doc in documents:
            doc.metadata["source"] = str(file_path)
            doc.metadata["file_type"] = "text"
        return documents

    def load_from_url(self, url: str) -> List[Document]:
        """
        从URL加载文档

        Args:
            url: 文档URL

        Returns:
            文档列表
        """
        # TODO: 实现URL文档加载
        raise NotImplementedError("URL文档加载功能开发中")
