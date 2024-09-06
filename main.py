import re
import os
import docx
from PyPDF2 import PdfReader
from lovins import stem
from gensim import corpora
from gensim.models import LsiModel
from gensim.similarities import MatrixSimilarity

def stem_text(text):
    stemmed_text = ''
    words = re.findall(r'\b\w+\b', text.lower())
    original_text = ' '.join(words)

    for word in words:
        stemmed_word = stem(word)
        stemmed_text += stemmed_word + ' '

    return original_text, stemmed_text

def count_words(text):
    word_set = set()
    word_count = {}

    words = re.findall(r'\b\w+\b', text.lower())

    for word in words:
        word_set.add(word)

    for unique_word in word_set:
        word_count[unique_word] = text.lower().count(unique_word)

    return word_count

def read_pdf(file_path):
    with open(file_path, 'rb') as file:
        reader = PdfReader(file)
        text = ''
        for page in reader.pages:
            text += page.extract_text()
        return text

def read_docx(file_path):
    doc = docx.Document(file_path)
    text = ''
    for paragraph in doc.paragraphs:
        text += paragraph.text
    return text

def read_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def read_file(file_path):
    separator = '\n\n--------------------\n\n'
    if file_path.lower().endswith('.pdf'):
        return read_pdf(file_path) + separator
    elif file_path.lower().endswith('.docx'):
        return read_docx(file_path) + separator
    else:
        return read_text(file_path) + separator

def find_files(directory):
    found_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            found_files.append(os.path.join(root, file))
    return found_files

def search_with_lsi(query, documents):

    tokenized_docs = [doc.lower().split() for doc in documents]
    dictionary = corpora.Dictionary(tokenized_docs)
    corpus = [dictionary.doc2bow(doc) for doc in tokenized_docs]

    lsi_model = LsiModel(corpus, id2word=dictionary, num_topics=10)
    index = MatrixSimilarity(lsi_model[corpus])

    query_bow = dictionary.doc2bow(query.lower().split())
    query_lsi = lsi_model[query_bow]

    sims = index[query_lsi]
    return sims

def main():
    documents = []
    while True:
        directory = input("Masukkan direktori tempat pencarian file atau ketik 'exit' untuk keluar: ")
        if directory.lower() == 'exit':
            print("Terima kasih!")
            break

        found_files = find_files(directory)
        if found_files:
            for file in found_files:
                content = read_file(file)
                original_content, stemmed_content = stem_text(content)
                word_count = count_words(content)

                print(f"\nFile: {file}")
                print("Jumlah kata pada file sebelum stemming:")
                for word, count in word_count.items():
                    print(f"{word}: {count}")

                print("\nKalimat sebelum stemming:")
                print(original_content)
                print("\nKalimat setelah stemming:")
                print(stemmed_content)
                print('-------------------------------------------------------------')
                documents.append(stemmed_content)

            query = input("Masukkan kata kunci untuk temu balik: ")

            similarities_lsi = search_with_lsi(query, documents)

            file_scores_lsi = {file: score for file, score in zip(found_files, similarities_lsi)}
            sorted_files_lsi = sorted(file_scores_lsi.items(), key=lambda x: x[1], reverse=True)

            print("\nScores for each document (LSI):")
            for file, score in sorted_files_lsi:
                print(f"{file}: Score - {score}")

            # Filtering relevant documents
            relevant_files_lsi = [file for file, score in sorted_files_lsi if score > 0]

            if relevant_files_lsi:
                print("\nKata kunci berada pada file (LSI):")
                for index, file in enumerate(relevant_files_lsi, start=1):
                    print(f"{index}. {file}")
            else:
                print("Kata kunci tidak ditemukan dengan metode LSI.")

        else:
            print("Direktori tidak berisi file.")

if __name__ == "__main__":
    main()
