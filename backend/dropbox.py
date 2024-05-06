import getpass
import os
import shutil

import dropbox
from dropbox.paper import ExportFormat, ListPaperDocsFilterBy
from settings import settings

# Notes: include document id in the file name for linking to URL

dbx = dropbox.Dropbox(getpass.getpass("Dropbox API Key:"))

def create_download_folder():
    if not os.path.exists(settings.download_folder):
        os.makedirs(settings.download_folder)

def clear_downloads_folder():
    for filename in os.listdir(settings.download_folder):
        file_path = os.path.join(settings.download_folder, filename)
        print(file_path)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

def get_file_path(doc_title):
    return os.path.join(settings.download_folder, f"{doc_title}.md")

def download_doc(doc_id, doc_title):
    result = dbx.paper_docs_download_to_file(get_file_path(doc_title), doc_id, ExportFormat('markdown'))
    print(f"- downloaded '{result.title}'")
    return result

def download_docs():
    print("Retrieving document IDs")
    doc_ids = dbx.paper_docs_list(filter_by=ListPaperDocsFilterBy.docs_created).doc_ids
    print(f"- {len(doc_ids)} documents found")

    print("Filtering documents in folder")
    docs_ids_in_folder = [doc_id for doc_id in doc_ids if settings.dropbox_remote_folder in [folder.name for folder in dbx.paper_docs_get_folder_info(doc_id).folders or []]]
    print(f"- {len(docs_ids_in_folder)} documents found in folder")

    print("Retrieving document titles")
    doc_titles = {doc_id: dbx.paper_docs_download(doc_id, ExportFormat('markdown'))[0].title for doc_id in docs_ids_in_folder}

    print("Downloading documents")
    results = [download_doc(doc_id, doc_titles[doc_id]) for doc_id in docs_ids_in_folder]
    print("Download complete")
    return results
