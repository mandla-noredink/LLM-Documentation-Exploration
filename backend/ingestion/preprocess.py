import re
import string

from langchain_core.documents import Document
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
# from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Do this first, that'll do something eval() 
# to "materialize" the LazyCorpusLoader
# https://stackoverflow.com/a/50614899
next(wn.all_synsets()) 


def pre_embedding_process(text: str) -> str:
    # Tokenization
    cleaned_tokens = re.sub(r"[_\-]", " ", text)
    tokens = word_tokenize(text)

    # noise removal
    cleaned_tokens = [re.sub(r"[^\w\s]", "", token) for token in tokens]
    cleaned_tokens = [re.sub(r"[\d]", "", token) for token in cleaned_tokens]
    cleaned_tokens = filter(
        lambda x: x not in string.punctuation + string.whitespace, cleaned_tokens
    )

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
