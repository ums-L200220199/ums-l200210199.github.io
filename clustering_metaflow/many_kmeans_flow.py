from metaflow import FlowSpec, step, Parameter, resources, conda_base, profile

# Mengatur environment Conda untuk Metaflow
@conda_base(python='3.8.3', libraries={'scikit-learn': '0.24.1'})
class ManyKmeansFlow(FlowSpec):
    # Parameter input yang dapat diubah saat menjalankan alur
    num_docs = Parameter('num-docs', help='Number of documents', default=1000)

    # Step 1: Inisialisasi data
    @resources(memory=4000)  # Mengalokasikan 4GB memori untuk langkah ini
    @step
    def start(self):
        import scale_data  # Asumsinya adalah modul untuk pengolahan data
        # Memuat dataset Yelp Reviews dan memprosesnya
        data = scale_data.load_yelp_reviews(self.num_docs)
        self.X, self.cols = scale_data.make_matrix(data)
        # Menentukan nilai K untuk K-Means
        self.k_params = list(range(5, 20, 5))  # Menggunakan nilai K dari 5 hingga 50 dengan interval 5
        self.next(self.train_kmeans, foreach='k_params')  # Melakukan foreach untuk setiap nilai K

    # Step 2: Melatih model K-Means
    @resources(cpu=4, memory=4000)  # Mengalokasikan 4 CPU dan 4GB memori
    @step
    def train_kmeans(self):
        from sklearn.cluster import KMeans
        # Mengambil nilai K dari parameter foreach
        self.k = self.input
        # Inisialisasi dan melatih model K-Means
        kmeans = KMeans(n_clusters=self.k, verbose=1, n_init=1)
        kmeans.fit(self.X)  # Memasukkan data untuk clustering
        self.clusters = kmeans.labels_  # Menyimpan hasil clustering
        self.next(self.analyze)

    # Step 3: Analisis hasil clustering
    @step
    def analyze(self):
        from analyze_kmeans import top_words
        # Menganalisis kata-kata teratas untuk setiap cluster
        self.result = top_words(self.k, self.clusters, self.X, self.cols)
        self.next(self.join)

    # Step 4: Menggabungkan hasil dari setiap foreach
    @step
    def join(self, inputs):
        # Menggabungkan semua hasil dari foreach (untuk setiap nilai K)
        self.results = [inp.result for inp in inputs]
        self.next(self.end)

    # Step 5: Menyelesaikan alur kerja
    @step
    def end(self):
        print("Workflow selesai.")

# Menjalankan alur kerja
if __name__ == '__main__':
    ManyKmeansFlow()
