import re
import string
# from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from langchain_core.documents import Document

stop_words = {
    "himself",
    "ve",
    "their",
    "ain",
    "needn",
    "she's",
    "yourself",
    "was",
    "an",
    "that'll",
    "wasn",
    "to",
    "hers",
    "y",
    "itself",
    "you're",
    "you",
    "how",
    "your",
    "m",
    "but",
    "ourselves",
    "him",
    "o",
    "a",
    "me",
    "as",
    "his",
    "our",
    "isn",
    "needn't",
    "weren",
    "what",
    "shan",
    "shan't",
    "had",
    "whom",
    "mightn",
    "who",
    "and",
    "of",
    "aren't",
    "i",
    "some",
    "isn't",
    "all",
    "yourselves",
    "those",
    "you'll",
    "you've",
    "these",
    "that",
    "such",
    "don",
    "they",
    "the",
    "it's",
    "is",
    "why",
    "just",
    "can",
    "theirs",
    "won",
    "do",
    "it",
    "he",
    "there",
    "hadn",
    "for",
    "or",
    "then",
    "no",
    "s",
    "ma",
    "hadn't",
    "so",
    "having",
    "she",
    "been",
    "we",
    "by",
    "my",
    "be",
    "yours",
    "ll",
    "nor",
    "them",
    "here",
    "than",
    "d",
    "being",
    "herself",
    "this",
    "very",
    "too",
    "both",
    "don't",
    "if",
    "you'd",
    "themselves",
    "am",
    "its",
    "her",
    "t",
    "are",
    "ours",
    "myself",
}


def pre_embedding_process(text: str) -> str:
    # Tokenization
    cleaned_tokens = re.sub(r"[_\-]", " ", text) 
    tokens = word_tokenize(text)

    # noise removal
    cleaned_tokens = [re.sub(r"[^\w\s]", "", token) for token in tokens]
    cleaned_tokens = [re.sub(r"[\d]", "", token) for token in cleaned_tokens]
    cleaned_tokens = filter(lambda x: x not in string.punctuation + string.whitespace, cleaned_tokens)

    # normalization
    cleaned_tokens = [token.lower() for token in cleaned_tokens]

    # stopword removal
    # NOTE: https://www.reddit.com/r/LocalLLaMA/comments/18xfr5g/comment/kg3wxnj/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button
    # stop_words = set(stopwords.words("english"))
    # cleaned_tokens = [token for token in cleaned_tokens if token not in stop_words]

    # lemmatization
    lemmatizer = WordNetLemmatizer()
    cleaned_tokens = [lemmatizer.lemmatize(token) for token in cleaned_tokens]

    return " ".join(cleaned_tokens)


def preprocess_document(document: Document) -> Document:
    document_copy = document.copy()
    document_copy.page_content = pre_embedding_process(document.page_content)
    return document_copy

