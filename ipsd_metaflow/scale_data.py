import tarfile
from itertools import islice
from metaflow import S3

def load_yelp_reviews(num_docs):
    """
    Fungsi untuk memuat ulasan Yelp dari file tar.gz yang disimpan di S3.
    
    Parameters:
        num_docs (int): Jumlah dokumen yang ingin dimuat.
    
    Returns:
        list: Daftar ulasan dalam jumlah tertentu.
    """
    with S3() as s3:
        # Mengambil file Yelp review dari S3
        res = s3.get('s3://fast-ai-nlp/yelp_review_full_csv.tgz')
        with tarfile.open(res.path) as tar:
            # Ekstraksi file CSV dari arsip
            datafile = tar.extractfile('cleaned_data_group.csv')
            # Mengembalikan sejumlah dokumen yang diminta
            return list(islice(datafile, num_docs))

def make_matrix(docs, binary=False):
    """
    Fungsi untuk mengubah dokumen menjadi matriks fitur dengan CountVectorizer.
    
    Parameters:
        docs (iterable): Koleksi dokumen teks.
        binary (bool): Jika True, hanya menghitung keberadaan kata (bukan frekuensi).
    
    Returns:
        tuple: Matriks fitur dan daftar kata.
    """
    from sklearn.feature_extraction.text import CountVectorizer
    
    # Inisialisasi CountVectorizer dengan parameter
    vec = CountVectorizer(min_df=10, max_df=0.1, binary=binary)
    # Membangun matriks fitur berdasarkan dokumen
    mtx = vec.fit_transform(docs)
    
    # Membuat daftar kolom (kata) berdasarkan vocab
    cols = [None] * len(vec.vocabulary_)
    for word, idx in vec.vocabulary_.items():
        cols[idx] = word
    
    return mtx, cols