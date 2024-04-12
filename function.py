# def translate_text(text, target_language='id'):
#     translator = Translator()
#     translated_text = translator.translate(text, dest=target_language)
#     return translated_text.text

# def translate_list_of_texts(text_list, target_language='id'):
#     translator = Translator()
#     translated_texts = [translator.translate(text, dest=target_language).text for text in text_list]
#     return translated_texts

# def display_user_rating_info(user_rating_type, translate):
#     if user_rating_type == "Low":
#         st.subheader("Low User Rating")
#         low_info = [
#             "- Applications with low user ratings typically receive ratings below 2.5.",
#             "- Users tend to be dissatisfied with these applications and may experience serious issues or lack of features.",
#             "- Developers need to pay attention to user reviews and make necessary improvements to enhance the application's quality."
#         ]
#         if translate:
#             low_info = translate_list_of_texts(low_info)
#         st.markdown("\n".join(low_info))
#     elif user_rating_type == "Medium":
#         st.subheader("Medium User Rating")
#         medium_info = [
#             "- Applications with medium user ratings typically receive ratings between 2.5 and 4.0.",
#             "- Users may have a reasonably good experience with these applications, but there is still room for improvement.",
#             "- Developers can review user reviews to identify areas for improvement and enhance user satisfaction."
#         ]
#         if translate:
#             medium_info = translate_list_of_texts(medium_info)
#         st.markdown("\n".join(medium_info))
#     elif user_rating_type == "High":
#         st.subheader("High User Rating")
#         high_info = [
#             "- Applications with high user ratings typically receive ratings between 4.0 and 5.0.",
#             "- Users are highly satisfied with these applications and tend to recommend them to others.",
#             "- Developers can leverage positive reviews to further promote the application and maintain user satisfaction."
#         ]
#         if translate:
#             high_info = translate_list_of_texts(high_info)
#         st.markdown("\n".join(high_info))

# # Fungsi untuk mengubah teks ke bahasa Indonesia
# def translate_list_of_texts(texts, target_language='id'):
#     translated_texts = []
#     for text in texts:
#         translated_texts.append(translate_text(text, target_language))
#     return translated_texts

# def predict():
#     size_bytes = st.number_input('Size Bytes', 0, 100)
#     price = st.number_input('Price', 0, 100)
#     rating_count_tot = st.number_input('Rating Count Total', 0, 100)
#     rating_count_ver = st.number_input('Rating Count Version', 0, 100)
#     user_rating_ver = st.number_input('User Rating Version', 0, 5.0)
#     sup_devices_num = st.number_input('Supported Devices Number', 0, 100)
#     lang_num = st.number_input('Language Number', 0, 100)
    
#     button = st.button('Predict')
#     if button:
#         data = pd.DataFrame({
#             'size_bytes': [size_bytes],
#             'price': [price],
#             'rating_count_tot': [rating_count_tot],
#             'rating_count_ver': [rating_count_ver],
#             'user_rating_ver': [user_rating_ver],
#             'sup_devices.num': [sup_devices_num],
#             'lang.num': [lang_num]
#         })
#         with open('gnb.pkl', 'rb') as file:
#             loaded_model = pickle.load(file)
#         predicted = loaded_model.predict(data)
#         if predicted == 1:
#             st.write('Low User Rating')
#         elif predicted == 2:
#             st.write('Medium User Rating')
#         elif predicted == 3:
#             st.write('High User Rating')
#         else:
#             st.error('Not Defined')
