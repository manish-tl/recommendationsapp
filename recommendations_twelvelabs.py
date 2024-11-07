import streamlit as st
from pinecone import Pinecone
from PIL import Image
import requests
from io import BytesIO

# Pinecone and dataset configurations
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
INDEX_NAME = 'recommendationsapp'
VIDEO_BASE_URL = "https://recsys.westlake.edu.cn/MicroLens-50k-Dataset/MicroLens-50k_videos/"
THUMBNAIL_BASE_URL = "https://recsys.westlake.edu.cn/MicroLens-50k-Dataset/MicroLens-50k_covers/"

# Initialize Pinecone
pinecone = Pinecone(api_key=PINECONE_API_KEY)
index = pinecone.Index(INDEX_NAME)

@st.cache_data
def get_all_videos():
    """Fetches all videos with embedding_scope=video from Pinecone and constructs URLs."""
    video_data = []
    top_k = 1000
    zero_vector = [0] * 1024
    results = index.query(
        vector=zero_vector,
        filter={'embedding_scope': 'video'}, top_k=top_k, include_values=False, include_metadata=True
    ).to_dict()
    
    for match in results['matches']:
        metadata = match.get('metadata', {})
        filename = metadata.get('filename')
        if filename:
            video_url = f"{VIDEO_BASE_URL}{filename}"
            thumbnail_url = f"{THUMBNAIL_BASE_URL}{filename.replace('.mp4', '.jpg')}"
            video_data.append({'id': match['id'], 'filename': filename, 'video_url': video_url, 'thumbnail_url': thumbnail_url})
    return video_data

def get_recommendations(video_id, exclude_filename):
    """Fetches top 10 similar videos with embedding_scope=video, excluding the original video filename."""
    results = index.query(
        id=video_id,
        top_k=10,
        filter={'embedding_scope': 'video'},
        include_values=False,
        include_metadata=True
    ).to_dict()
    
    recommendations = []
    for match in results['matches']:
        metadata = match.get('metadata', {})
        filename = metadata.get('filename')
        if filename and filename != exclude_filename:  # Exclude videos with the same filename
            video_url = f"{VIDEO_BASE_URL}{filename}"
            thumbnail_url = f"{THUMBNAIL_BASE_URL}{filename.replace('.mp4', '.jpg')}"
            recommendations.append({'id': match['id'], 'filename': filename, 'video_url': video_url, 'thumbnail_url': thumbnail_url})
    return recommendations

def display_videos(video_data):
    """Displays videos with thumbnails in a grid layout with Recommend option."""
    st.markdown("### Explore Videos")
    st.markdown("Discover new content by clicking **Recommend Videos** below each thumbnail.")
    
    cols = st.columns(3)
    for i, video in enumerate(video_data):
        with cols[i % 3]:
            st.image(video['thumbnail_url'], use_column_width=True)
            if st.button("Recommend Videos", key=video['id']):
                # Set session state and force rerun
                st.session_state["selected_video_id"] = video['id']
                st.session_state["selected_filename"] = video['filename']
                st.session_state["selected_video_url"] = video['video_url']
                st.session_state["selected_thumbnail_url"] = video['thumbnail_url']
                st.session_state["page"] = "recommendations"
                st.experimental_rerun()  # Force rerun to apply state changes immediately

def display_recommendations(recommendations):
    """Displays the original video and recommended videos on a new page, with nested recommend options."""
    st.subheader("Original Video")
    st.video(st.session_state["selected_video_url"])
    
    st.subheader("Recommended Videos")
    cols = st.columns(2)
    for i, rec in enumerate(recommendations):
        with cols[i % 2]:
            st.video(rec['video_url'], start_time=0)
            if st.button("Recommend Videos", key=rec['id']):
                # Update session state for the new selected video and rerun
                st.session_state["selected_video_id"] = rec['id']
                st.session_state["selected_filename"] = rec['filename']
                st.session_state["selected_video_url"] = rec['video_url']
                st.session_state["selected_thumbnail_url"] = rec['thumbnail_url']
                st.experimental_rerun()

# Streamlit App Layout
st.set_page_config(page_title="Twelve Labs Video Recommendations", layout="centered")

# Display the logo and title
st.image("https://cdn.prod.website-files.com/63d42c9fdd5148cd77b8f0c6/63d43a258f052713592ddd16_twelvelabs_logo_black%202.svg", width=200)  # Adjust width as needed
st.title("Recommendations using Twelve Labs Multimodal Embeddings")
st.markdown("Welcome to the video recommendation app! Start exploring videos and discovering similar content powered by **Twelve Labs Multimodal Embeddings**.")

# Check session state for page navigation
if "page" not in st.session_state:
    st.session_state["page"] = "main"

if st.session_state["page"] == "recommendations" and "selected_video_id" in st.session_state:
    video_id = st.session_state["selected_video_id"]
    exclude_filename = st.session_state.get("selected_filename", "")
    recommendations = get_recommendations(video_id, exclude_filename)
    display_recommendations(recommendations)
    if st.button("Back to Main"):
        st.session_state["page"] = "main"
else:
    # Fetch and display all videos on the main page
    video_data = get_all_videos()
    if video_data:
        display_videos(video_data)
    else:
        st.write("No videos found with the specified embedding scope.")
