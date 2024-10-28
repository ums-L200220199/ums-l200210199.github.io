class Mahasiswa:
    def __init__(self, nama, nim):
        self.nama = nama
        self.nim = nim
        self.spp_terbayar = False
        self.kehadiran = 0
        self.tugas = []
        self.uts = None
        self.uas = None
        self.nilai_akhir = None

    def bayar_spp(self):
        print(f"{self.nama} sedang membayar SPP...")
        self.spp_terbayar = True
        print("SPP telah dibayar.")

    def ikuti_kuliah(self, jumlah_pertemuan):
        if not self.spp_terbayar:
            print("Harap bayar SPP terlebih dahulu!")
            return

        for i in range(1, jumlah_pertemuan + 1):
            print(f"{self.nama} menghadiri kuliah ke-{i}")
            self.kehadiran += 1

    def kumpul_tugas(self, nilai_tugas):
        if not self.spp_terbayar:
            print("Harap bayar SPP terlebih dahulu!")
            return

        print(f"{self.nama} mengumpulkan tugas dengan nilai {nilai_tugas}")
        self.tugas.append(nilai_tugas)

    def ujian_uts(self, nilai):
        if not self.spp_terbayar:
            print("Harap bayar SPP terlebih dahulu!")
            return

        print(f"{self.nama} mengikuti UTS dengan nilai {nilai}")
        self.uts = nilai

    def ujian_uas(self, nilai):
        if not self.spp_terbayar:
            print("Harap bayar SPP terlebih dahulu!")
            return

        print(f"{self.nama} mengikuti UAS dengan nilai {nilai}")
        self.uas = nilai

    def hitung_nilai_akhir(self):
        if not self.spp_terbayar:
            print("Harap bayar SPP terlebih dahulu!")
            return

        if self.uts is None or self.uas is None or not self.tugas:
            print("Nilai belum lengkap. Pastikan UTS, UAS, dan tugas sudah ada.")
            return

        rata_tugas = sum(self.tugas) / len(self.tugas)
        self.nilai_akhir = 0.3 * rata_tugas + 0.3 * self.uts + 0.4 * self.uas
        print(f"Nilai akhir {self.nama} adalah {self.nilai_akhir:.2f}")

# Simulasi Proses Kuliah
mahasiswa1 = Mahasiswa("Dicky", "12345678")

# Bayar SPP
mahasiswa1.bayar_spp()

# Mengikuti kuliah sebanyak 10 pertemuan
mahasiswa1.ikuti_kuliah(10)

# Mengumpulkan 3 tugas
mahasiswa1.kumpul_tugas(80)
mahasiswa1.kumpul_tugas(85)
mahasiswa1.kumpul_tugas(90)

# Mengikuti UTS dan UAS
mahasiswa1.ujian_uts(88)
mahasiswa1.ujian_uas(92)

# Menghitung nilai akhir
# test
mahasiswa1.hitung_nilai_akhir()
