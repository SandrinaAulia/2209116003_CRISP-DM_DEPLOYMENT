import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import altair as alt
from googletrans import Translator
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

st.title('Mobile App Store (7200 apps) Analytics for Mobile Apps')
# Menampilkan foto di bagian bawah
st.image('About.jpg', caption='About Mobile App Store', use_column_width=True)

URL = 'Data Cleaned.csv'
df = pd.read_csv(URL)

file_path_clf = 'gnb.pkl'
with open(file_path_clf, 'rb') as f:
    clf = joblib.load(f)

url_data_clean_before_mapping = 'https://raw.githubusercontent.com/SandrinaAulia/MiniProject-DataMining/main/AppleStore.csv'
dfa = pd.read_csv(url_data_clean_before_mapping)

selected_page = st.sidebar.selectbox(
    'Select Page',
    ['Introducing','Data Distribution','Relationship Analysis','Composition & Comparison','Predict','Clustering']
)

if selected_page == 'Introducing':
    data = pd.read_csv(URL)
    
    st.subheader("Mobile App Store")
    st.write("""
    Statistik Mobile Apps (Mobile Apple Store / iOS)
    Dunia aplikasi ponsel selalu berubah dan penuh tantangan. Penggunaan ponsel lebih banyak daripada penggunaan desktop, dan Android menguasai sekitar 53.2% pasar ponsel pintar, 
    sementara iOS sekitar 43%. Untuk mendapatkan lebih banyak pengunduh aplikasi, penting bagi Anda untuk memastikan bahwa aplikasi Anda mudah ditemukan. 
    Melalui analisis aplikasi ponsel, Anda bisa memahami strategi yang sudah ada untuk meningkatkan jumlah pengguna di masa depan.
    Saat ini, ada jutaan aplikasi yang tersedia, dan dataset berikut ini sangat penting untuk mengetahui aplikasi-aplikasi teratas di toko aplikasi iOS. 
    Dataset ini berisi lebih dari 7000 detail aplikasi mobile Apple iOS. 
    Data tersebut diambil dari iTunes Search API di situs web Apple Inc. Menggunakan alat web scraping R dan Linux untuk melakukan penelitian ini.
    """)
    st.markdown("[Sumber : Kaggle.com](https://www.kaggle.com/datasets/ramamet4/app-store-apple-data-set-10k-apps/data)")
    st.title("User Rating Information")
    st.write("""
        Prediksi user rating penting untuk mengamankan kesuksesan aplikasi di pasar yang kompetitif. Dengan memahami tren pasar dan keinginan pengguna, pengembang dapat meningkatkan kualitas aplikasi dan strategi pemasaran. 
        Analisis ulasan atau rating pengguna membantu dalam memperkirakan penerimaan aplikasi di pasar yang terus berkembang. Dengan demikian, prediksi user rating berperan penting dalam menghadapi tantangan dan memastikan aplikasi 
        tetap relevan di lingkungan yang dinamis.
        Pilihlah jenis kategori untuk user rating dibawah ini : """)

    def translate_text(text, target_language='id'):
        translator = Translator()
        translated_text = translator.translate(text, dest=target_language)
        return translated_text.text

    def translate_list_of_texts(text_list, target_language='id'):
        translator = Translator()
        translated_texts = [translator.translate(text, dest=target_language).text for text in text_list]
        return translated_texts

    def display_user_rating_info(user_rating_type, translate):
        if user_rating_type == "Low":
            st.subheader("Low User Rating")
            low_info = [
                "- Applications with low user ratings typically receive ratings below 2.5.",
                "- Users tend to be dissatisfied with these applications and may experience serious issues or lack of features.",
                "- Developers need to pay attention to user reviews and make necessary improvements to enhance the application's quality."
            ]
            if translate:
                low_info = translate_list_of_texts(low_info)
            st.markdown("\n".join(low_info))
        elif user_rating_type == "Medium":
            st.subheader("Medium User Rating")
            medium_info = [
                "- Applications with medium user ratings typically receive ratings between 2.5 and 4.0.",
                "- Users may have a reasonably good experience with these applications, but there is still room for improvement.",
                "- Developers can review user reviews to identify areas for improvement and enhance user satisfaction."
            ]
            if translate:
                medium_info = translate_list_of_texts(medium_info)
            st.markdown("\n".join(medium_info))
        elif user_rating_type == "High":
            st.subheader("High User Rating")
            high_info = [
                "- Applications with high user ratings typically receive ratings between 4.0 and 5.0.",
                "- Users are highly satisfied with these applications and tend to recommend them to others.",
                "- Developers can leverage positive reviews to further promote the application and maintain user satisfaction."
            ]
            if translate:
                high_info = translate_list_of_texts(high_info)
            st.markdown("\n".join(high_info))
    
    user_rating_types = ["Low", "Medium", "High"]
    user_rating_type = st.selectbox("User Rating Type", user_rating_types)
    
    translate = st.checkbox("Translate to Bahasa Indonesia")

    display_user_rating_info(user_rating_type, translate)
    

    st.title('Mobile App Store Dataset')
    st.write('Berikut merupakan tampilan dataset setelah di cleaned')
    URL = 'Data Cleaned.csv'
    df = pd.read_csv(URL)
    st.write(df)

    def categorize_Urating(score):
        if score < 2.5:
            return 'Rendah'
        elif 2.5 <= score < 4.0:
            return 'Sedang'
        elif 4.0 <= score <= 5.0:
            return 'Tinggi'
        else:
            return 'Not Valid Score'

    df['user_rating'] = df['user_rating'].apply(categorize_Urating)
    df['user_rating'] = df['user_rating'].map({'Rendah': 1, 'Sedang': 2, 'Tinggi': 3}).astype(int)


elif selected_page == "Data Distribution":
    data = pd.read_csv(url_data_clean_before_mapping)
    st.subheader("Data Distribution Section")

    feature_options = ['Jumlah Aplikasi berdasarkan Kategori Ukuran Aplikasi', 'Korelasi antara User Rating dan Kategori Ukuran Aplikasi', 'Jumlah Aplikasi Berdasarkan Kategori Genre']
    selected_feature = st.selectbox('Select Feature', feature_options)

    if selected_feature == 'Jumlah Aplikasi berdasarkan Kategori Ukuran Aplikasi':
        # Konversi ukuran dari bytes ke Mega Bytes
        dfa['size_mb'] = dfa['size_bytes'] / 1000000
        summary_stats = dfa['size_mb'].describe() # Menampilkan statistik
        print(summary_stats)

        # Pembuatan kolom baru 'SizeCatg atau Size Category' dan inisialisasi dengan ukuran aplikasi dalam string
        dfa['SizeCatg'] = dfa['size_mb'].astype(str)

        # Proses pengklasifikasian aplikasi berdasarkan ukuran
        for i in range(len(dfa['SizeCatg'])):
            if dfa.loc[i, 'size_mb'] < 100:
                dfa.loc[i, 'SizeCatg'] = 'Small Size App'
            elif dfa.loc[i, 'size_mb'] >= 100 and dfa.loc[i, 'size_mb'] < 300:
                dfa.loc[i, 'SizeCatg'] = 'Medium Size App'
            elif dfa.loc[i, 'size_mb'] >= 300 and dfa.loc[i, 'size_mb'] <= 5000:
                dfa.loc[i, 'SizeCatg'] = 'Big Size App'

        # Mengubah klasifikasi ke format category
        dfa['SizeCatg'] = pd.Categorical(dfa['SizeCatg'], categories=['Small Size App', 'Medium Size App', 'Big Size App'], ordered=True)

        # Total masing-masing kategori ukuran aplikasi
        summary_stats = dfa['SizeCatg'].value_counts()

        print(summary_stats)
        # Menampilkan plot
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.barplot(x=summary_stats.index, y=summary_stats.values, hue=summary_stats.index, palette="BuGn", dodge=False, ax=ax)
        plt.title('Jumlah Aplikasi iOS App Store Berdasarkan Kategori Ukuran')
        plt.xlabel('Kategori Ukuran Aplikasi')
        plt.ylabel('Jumlah')
        plt.legend(title='Kategori Ukuran Aplikasi', bbox_to_anchor=(1.05, 1), loc='upper left')
        st.pyplot(fig)

        st.write("Interpretasi: Grafik jumlah aplikasi iOS App Store berdasarkan kategori ukuran menunjukkan bahwa mayoritas aplikasi termasuk dalam kategori ukuran sedang, diikuti oleh kategori ukuran besar, dan jumlah aplikasi dengan ukuran kecil paling sedikit.")
        st.write("Insight: Analisis distribusi ukuran aplikasi ini mengungkapkan bahwa sebagian besar pengembang aplikasi cenderung mengembangkan aplikasi dengan ukuran sedang, yang mungkin mencerminkan keseimbangan antara fitur dan penggunaan memori.")
        st.write(" Actionable Insight: Untuk menarik lebih banyak pengguna dan memperluas pangsa pasar, pengembang dapat mempertimbangkan untuk mengembangkan lebih banyak aplikasi dengan ukuran kecil atau menengah. Menyediakan aplikasi dengan ukuran yang lebih kecil dapat meningkatkan daya tarik aplikasi, terutama bagi pengguna yang memiliki keterbatasan ruang penyimpanan di perangkat mereka.")

    elif selected_feature == 'Korelasi antara User Rating dan Kategori Ukuran Aplikasi':
        fig = px.box(df, x='SizeCatg', y='user_rating_ver',
            title='Persebaran Rating Berdasarkan Ukuran Aplikasi',
            labels={'SizeCatg': 'Kategori Ukuran Aplikasi', 'user_rating_ver': 'User Rating (Max = 5)'},
            template='plotly_white')
        fig.show()
    
    elif selected_feature == 'Jumlah Aplikasi Berdasarkan Kategori Genre':
    # Analisis Data Menggunakan Count Plot untuk Jumlah Aplikasi Berdasarkan Kategori Genre
        color_palette = 'summer'

        plt.figure(figsize=(10, 6))
        ax = sns.countplot(y='prime_genre', data=dfa, palette=color_palette, hue='prime_genre', order=dfa['prime_genre'].value_counts().index)

        for p in ax.patches:
            ax.annotate(f'{p.get_width()}', (p.get_width() + 0.1, p.get_y() + p.get_height() / 2), ha='left', va='center')

        plt.title('Jumlah Aplikasi iOS App Store Berdasarkan Kategori', fontsize=16)
        plt.xlabel('Jumlah', fontsize=12)
        plt.ylabel('Genre Aplikasi', fontsize=12)
        st.pyplot(plt)

        st.write("Interpretasi: Grafik jumlah aplikasi iOS App Store berdasarkan kategori genre menunjukkan bahwa genre Games mendominasi jumlah aplikasi yang tersedia, diikuti oleh genre Entertainment dan Education.")
        st.write("Insight: Analisis distribusi genre aplikasi ini mengungkapkan bahwa mayoritas pengembang cenderung fokus pada pengembangan game, yang mungkin mencerminkan popularitas dan permintaan pasar yang tinggi untuk aplikasi permainan.")
        st.write("Actionable Insight: Untuk pengembang yang ingin memasuki pasar aplikasi dengan lebih banyak variasi, mereka dapat mempertimbangkan untuk mengembangkan aplikasi dalam genre yang kurang populer seperti Medical, Catalogs, atau Navigation. Dengan demikian, mereka dapat mengurangi persaingan dan memperluas pangsa pasar mereka.")

elif selected_page == "Relationship Analysis":
    data = pd.read_csv(url_data_clean_before_mapping )
    st.subheader("Relationship Analysis Section")
    numeric_data = data.select_dtypes(include='number')
    # Korelasi Menggunakan Heatmap/Matriks Plot
    sns.heatmap(dfa.corr(numeric_only=True), annot=True, fmt=".2f", cbar=True, cmap="Greens")
    plt.gcf().set_size_inches(15, 10)
    st.pyplot(plt.gcf())
    st.write("Interpretasi: Heatmap korelasi menampilkan hubungan antara fitur-fitur numerik dalam dataset. Nilai korelasi berkisar dari -1 hingga 1, dengan 1 menunjukkan korelasi positif sempurna, -1 menunjukkan korelasi negatif sempurna, dan 0 menunjukkan tidak adanya korelasi antara dua fitur.")
    st.write("Insight: Dari visualisasi ini, kita dapat melihat seberapa kuat hubungan linier antara berbagai fitur numerik dalam dataset. Fitur-fitur dengan korelasi positif yang tinggi menunjukkan bahwa mereka cenderung bergerak bersama-sama, sedangkan korelasi negatif menunjukkan bahwa mereka cenderung bergerak ke arah yang berlawanan.")
    st.write("Actionable Insight: Dengan memahami korelasi antara fitur-fitur, kita dapat mengidentifikasi fitur-fitur yang saling berhubungan dan memahami bagaimana mereka berinteraksi dalam dataset. Informasi ini dapat digunakan untuk pengembangan model, pengoptimalan fitur, atau pengambilan keputusan lainnya dalam analisis data. Misalnya, jika dua fitur memiliki korelasi yang tinggi, mungkin dapat dipertimbangkan untuk menggunakan salah satunya saja dalam model untuk menghindari multicollinearity.")


elif selected_page == "Composition & Comparison":
    data = pd.read_csv(URL)
    
    # Definisikan fungsi kategori dan aplikasikan pada kolom 'user_rating'
    def categorize_Urating(score):
        if score < 2.5:
            return 'Rendah'
        elif 2.5 <= score < 4.0:
            return 'Sedang'
        elif 4.0 <= score <= 5.0:
            return 'Tinggi'
        else:
            return 'Not Valid Score'

    data['user_rating_cat'] = data['user_rating'].apply(categorize_Urating)

    # Hitung rata-rata fitur untuk setiap kategori peringkat pengguna (User Rating Category)
    user_rating_composition = data.groupby('user_rating_cat').mean()

    # Plot komposisi fitur untuk setiap kategori peringkat pengguna (User Rating Category)
    plt.figure(figsize=(10, 6))
    sns.heatmap(user_rating_composition.T, annot=True, cmap='YlGnBu')
    plt.title('Composition for each User Rating Category')
    plt.xlabel('User Rating Category')
    plt.ylabel('Feature')
    st.pyplot(plt)

    st.write("Interpretasi: Heatmap menunjukkan komposisi rata-rata fitur untuk setiap kategori peringkat pengguna (User Rating Category). Setiap baris pada heatmap mewakili satu fitur, sedangkan setiap kolom mewakili kategori peringkat pengguna. Nilai pada heatmap menunjukkan rata-rata nilai fitur untuk setiap kategori peringkat pengguna.")
    st.write("Insight: Dari visualisasi ini, kita dapat melihat bagaimana komposisi fitur berbeda-beda untuk setiap kategori peringkat pengguna. Fitur-fitur dengan nilai yang tinggi dalam kategori peringkat pengguna tertentu dapat dianggap sebagai fitur yang penting atau berpengaruh dalam menentukan peringkat pengguna.")
    st.write("Actionable Insight: Dengan memahami komposisi fitur untuk setiap kategori peringkat pengguna, kita dapat mengidentifikasi fitur-fitur yang paling memengaruhi peringkat pengguna. Hal ini dapat membantu dalam pengembangan strategi untuk meningkatkan peringkat pengguna, seperti fokus pada peningkatan fitur-fitur yang memiliki kontribusi besar terhadap peringkat pengguna yang tinggi.")


elif selected_page == "Predict":
    data = pd.read_csv(URL)
    st.subheader("Predicting User Rating")

    # Fungsi untuk kategorisasi user rating
    def categorize_Urating(score):
        if score < 2.5:
            return 'Rendah'
        elif 2.5 <= score < 4.0:
            return 'Sedang'
        elif 4.0 <= score <= 5.0:
            return 'Tinggi'
        else:
            return 'Not Valid Score'

    # Buat DataFrame df untuk simulasi
    df = pd.DataFrame(columns=['rating_count_tot', 'rating_count_ver', 'user_rating', 'user_rating_ver', 'prime_genre', 'sup_devices.num', 'lang.num', 'SizeCatg', 'PriceCatg'])
    df.loc[0] = [1000000, 1000000, 5.0, 5.0, 5, 100, 100, 3, 4]  # Contoh data

    # Fungsi untuk memprediksi user rating
    def predict():
        # Input nilai fitur dari pengguna
        rating_count_tot = st.number_input('Total Rating Count', 0, 100000)
        rating_count_ver = st.number_input('Rating Count Version', 0, 10000)
        user_rating_ver = st.number_input('User Rating Version', 0.0, 5.0)
        sup_devices_num = st.number_input('Supported Devices Number', 0, 1000)
        lang_num = st.number_input('Language Number', 0, 100)
        prime_genre = st.number_input('Prime Genre', 1, 5)
        size_catg = st.number_input('Size Category', 1, 3)
        price_catg = st.number_input('Price Category',1, 4)

        button = st.button('Predict')

        if button:
            data = pd.DataFrame({
                'rating_count_tot': [rating_count_tot],
                'rating_count_ver': [rating_count_ver],
                'user_rating_ver': [user_rating_ver],
                'sup_devices.num': [sup_devices_num],
                'lang.num': [lang_num],
                'prime_genre': [prime_genre],
                'SizeCatg': [size_catg],
                'PriceCatg': [price_catg]
            })

            # Load model yang sudah disimpan sebelumnya
            with open('gnb.pkl', 'rb') as file:
                loaded_model = pickle.load(file)

            # Lakukan prediksi
            predicted = loaded_model.predict(data)
        
            # Konversi prediksi menjadi kategori user_rating
            predicted_rating = categorize_Urating(predicted[0])

            # Tampilkan hasil prediksi
            st.write("Predicted User Rating:", predicted_rating)

    predict()

    st.write("Interpretasi: Pada halaman ini, pengguna dapat memasukkan nilai fitur dari aplikasi mobile dan memprediksi peringkat pengguna (user rating) yang mungkin didapatkan berdasarkan fitur-fitur tersebut. Setelah memasukkan nilai fitur, pengguna dapat menekan tombol Predict untuk melihat hasil prediksi.")
    st.write("Insight: Fitur-fitur yang dimasukkan oleh pengguna, seperti jumlah total rating, jumlah rating versi terbaru, peringkat pengguna versi terbaru, jumlah perangkat yang didukung, jumlah bahasa yang didukung, genre utama, kategori ukuran, dan kategori harga, digunakan untuk memprediksi peringkat pengguna. Prediksi dilakukan dengan menggunakan model yang telah dilatih sebelumnya.")
    st.write("Actionable Insight: Dengan menggunakan fitur ini, pengembang aplikasi dapat menguji berbagai kombinasi fitur dan melihat bagaimana perubahan pada fitur-fitur tersebut dapat memengaruhi prediksi peringkat pengguna. Hal ini dapat membantu dalam pengambilan keputusan terkait pengembangan dan pemasaran aplikasi, seperti menyesuaikan fitur-fitur untuk meningkatkan peringkat pengguna atau menentukan strategi harga berdasarkan prediksi peringkat pengguna yang dihasilkan.")


elif selected_page == "Clustering":
    data = pd.read_csv(URL)
    st.subheader("Clustering User Rating")

    # Fitur-fitur yang akan digunakan untuk clustering
    features = ['rating_count_tot', 'rating_count_ver', 'user_rating_ver', 
                'sup_devices.num', 'lang.num', 'prime_genre','SizeCatg', 'PriceCatg']

    # Memilih hanya fitur-fitur yang akan digunakan untuk clustering
    X = data[features]

    # Normalisasi data
    scaler = MinMaxScaler()
    X_norm = scaler.fit_transform(X)

    # Melakukan clustering dengan KMeans
    kmeans = KMeans(n_clusters=6, init='k-means++', algorithm='lloyd', random_state=42)
    kmeans.fit(X_norm)
    kmeans_clust = kmeans.predict(X_norm)

    # Menambahkan hasil clustering ke dalam DataFrame
    data['kmeans_cluster'] = kmeans_clust

    # Visualisasi hasil clustering (opsional)
    # (Silakan sesuaikan dengan metode visualisasi yang Anda inginkan)

    # Input fitur dari pengguna
    st.write("Masukkan Nilai Fitur untuk Melihat Klaster")
    rating_count_tot = st.number_input('Total Rating Count', 0, 100000)
    rating_count_ver = st.number_input('Rating Count Version', 0, 10000)
    user_rating_ver = st.number_input('User Rating Version', 0.0, 5.0)
    sup_devices_num = st.number_input('Supported Devices Number', 0, 1000)
    lang_num = st.number_input('Language Number', 0, 100)
    prime_genre = st.number_input('Prime Genre', 1,5)
    SizeCatg = st.number_input('Size Category',1,3)
    PriceCatg = st.number_input('Price Category',1,4)

    # Button untuk melakukan clustering
    button_cluster = st.button("Cluster")

    if button_cluster:
        # Lakukan prediksi klaster untuk nilai fitur yang dimasukkan pengguna
        user_input = [[rating_count_tot, rating_count_ver, user_rating_ver, 
                       sup_devices_num, lang_num, prime_genre, SizeCatg, PriceCatg]]
        user_input_norm = scaler.transform(user_input)
        predicted_cluster = kmeans.predict(user_input_norm)
        st.write("Hasil Klaster untuk Nilai Fitur yang Dimasukkan Pengguna:", predicted_cluster[0])

    # Visualisasi hasil clustering
    import matplotlib.pyplot as plt

    # Scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(data['rating_count_tot'], data['user_rating_ver'], c=data['kmeans_cluster'], cmap='viridis')
    plt.xlabel('Total Rating Count')
    plt.ylabel('User Rating Version')
    plt.title('Scatter Plot of K-Means Clustering')
    plt.colorbar(label='K-Means Cluster')
    plt.grid(True)
    st.pyplot(plt)

    st.write("Interpretasi: Pada halaman ini, dilakukan proses clustering terhadap data aplikasi mobile berdasarkan beberapa fitur seperti jumlah total rating, jumlah rating versi terbaru, peringkat pengguna versi terbaru, jumlah perangkat yang didukung, jumlah bahasa yang didukung, genre utama, kategori ukuran, dan kategori harga. Hasil clustering digunakan untuk mengelompokkan aplikasi ke dalam beberapa klaster berdasarkan kemiripan fitur-fitur tersebut.")
    st.write("Insight: Dengan melakukan clustering, aplikasi mobile dapat dikelompokkan ke dalam beberapa klaster berdasarkan fitur-fitur tertentu. Hal ini dapat memberikan wawasan tentang pola atau karakteristik yang mungkin ada di dalam data. Misalnya, aplikasi dengan fitur-fitur yang mirip akan dikelompokkan bersama dalam satu klaster, sementara aplikasi dengan fitur yang berbeda akan dikelompokkan ke dalam klaster yang berbeda pula.")
    st.write("Actionable Insight: Dari hasil clustering, pengembang aplikasi dapat memahami pola-pola yang ada di dalam data dan mengambil tindakan yang sesuai. Misalnya, mereka dapat menyesuaikan strategi pemasaran atau pengembangan aplikasi berdasarkan klaster-klasternya. Selain itu, pengembang juga dapat melakukan analisis lebih lanjut untuk memahami karakteristik setiap klaster dan menyesuaikan strategi mereka sesuai dengan kebutuhan dan preferensi pengguna dalam setiap klaster.")

# Fungsi untuk mengubah teks ke bahasa Indonesia
def translate_list_of_texts(texts, target_language='id'):
    translated_texts = []
    for text in texts:
        translated_texts.append(translate_text(text, target_language))
    return translated_texts