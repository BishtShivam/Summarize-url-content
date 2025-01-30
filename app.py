import validators
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader

# streamlit app

st.set_page_config(page_title="Langchain: Summarized Text From YT or Website", page_icon="🦜")
st.title("Summarize Text from a Website, using Langchain🦜")
st.subheader('Input URL to get a breif summary of the content in it ⬇️')

summary_length = st.slider("Select the length of summary", 50, 500, 300)
st.write(f"## Summary in {summary_length} words")

#Get the GROQ api key and url(YT or website) to summarize
with st.sidebar:
    groq_api_key=st.text_input("GROQ API Key",value="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", type="password")
    
generic_url=st.text_input("URL", label_visibility="collapsed")

#Gemma model using GROQ API
llm=ChatGroq(model="Gemma2-9b-It", groq_api_key=groq_api_key)

prompt_template=f"""
Provide a summary of the following content in {summary_length} words.
Content: {{text}}
"""

prompt=PromptTemplate(template=prompt_template, input_variables=["text"])

if st.button("Generate Summary"):
    # validate inputs
    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Please provide the innformation")
    elif not validators.url(generic_url):
        st.error("Please enter a valid url. It can may be a YouTube url or a Website")
    else:
        try:
            with st.spinner("Waiting..."):
                # laoding the website or youtube video data
                if "youtube.com" in generic_url:
                    loader=YoutubeLoader.from_youtube_url(generic_url, add_video_info=True)          
                else:
                    loader=UnstructuredURLLoader(urls=[generic_url], ssl_verify=False,
                                                headers={"User-Agent": "Morzila/5.0 (Macintosh; Intel Mac OS  13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"})
                    docs=loader.load()
                    
                    # Chain for summarization
                    chain=load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                    output_summary=chain.run(docs)
                    
                    st.success(output_summary)
                    
        except Exception as e:
            st.exception(f"Exception:{e}")
