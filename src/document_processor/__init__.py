"""
文档处理模块
"""
from .loader import DocumentLoader
from .splitter import LegalDocumentSplitter
from .web_crawler import WebCrawler

__all__ = ["DocumentLoader", "LegalDocumentSplitter", "WebCrawler"]
