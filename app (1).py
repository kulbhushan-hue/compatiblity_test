import streamlit as st
import pandas as pd
from io import BytesIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="z/OS 3.2 Compatibility Matrix", layout="wide")
st.title("z/OS 3.2 Compatibility Matrix Dashboard")


def generate_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="All Products")

        if 'R3.2 Compatible' in df.columns:
            df[df['R3.2 Compatible']=='Y'].to_excel(writer, sheet_name='Compatible', index=False)
            df[df['R3.2 Compatible']=='N'].to_excel(writer, sheet_name='Not Compatible', index=False)

        summary = df.groupby('Vendor').size().reset_index(name='Product Count')
        summary.to_excel(writer, sheet_name='Vendor Summary', index=False)

    output.seek(0)
    return output


def create_rag_index(df):
    data=df.fillna('').copy()
    data['combined']=data.astype(str).agg(' '.join, axis=1)
    vectorizer=TfidfVectorizer(stop_words='english')
    vectors=vectorizer.fit_transform(data['combined'])
    return vectorizer,vectors

uploaded_file = st.file_uploader('Upload Compatibility Excel File', type=['xlsx'])

if uploaded_file:
    df = pd.read_excel(uploaded_file)

    vendors = st.sidebar.multiselect('Vendor', sorted(df['Vendor'].dropna().unique()))
    compatibility = st.sidebar.selectbox('Compatibility', ['All','Y','N'])

    filtered_df=df.copy()

    if vendors:
        filtered_df=filtered_df[filtered_df['Vendor'].isin(vendors)]

    if compatibility!='All':
        filtered_df=filtered_df[filtered_df['R3.2 Compatible']==compatibility]

    st.subheader('Compatibility Matrix')
    st.dataframe(filtered_df,use_container_width=True)

    st.download_button(
        '📥 Download Compatibility Matrix',
        generate_excel(filtered_df),
        file_name='zos32_compatibility_matrix.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )

    c1,c2,c3,c4=st.columns(4)
    c1.metric('Products', len(filtered_df))
    c2.metric('Compatible', len(filtered_df[filtered_df['R3.2 Compatible']=='Y']))
    c3.metric('Not Compatible', len(filtered_df[filtered_df['R3.2 Compatible']=='N']))
    c4.metric('Vendors', filtered_df['Vendor'].nunique())

    st.subheader('RAG Search')
    query=st.text_input('Search Product / PTF / APAR / USERMOD')

    vectorizer,vectors=create_rag_index(df)

    if query:
        qv=vectorizer.transform([query])
        scores=cosine_similarity(qv,vectors).flatten()
        idx=scores.argsort()[::-1][:10]
        result=df.iloc[idx].copy()
        result['Score']=scores[idx]
        st.dataframe(result,use_container_width=True)
else:
    st.info('Upload compatibility Excel file')
