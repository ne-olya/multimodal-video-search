from pathlib import Path

import streamlit as st

from videoscope.encoders import OpenClipEncoder
from videoscope.retrieval import MultimodalSegmentIndex

st.set_page_config(page_title="VideoScope", layout="wide")
st.title("VideoScope")
artifact_dir = st.sidebar.text_input("Индекс", "artifacts/default")
mode = st.sidebar.segmented_control("Модальности", ["video", "video_speech", "all"], default="all")
query = st.text_input("Что происходит в видео?")
if query and Path(artifact_dir, "segments.npz").exists():
    index = MultimodalSegmentIndex.load(artifact_dir)
    encoder = OpenClipEncoder()
    weights = (
        {"visual": 1, "speech": 0, "ocr": 0}
        if mode == "video"
        else (
            {"visual": 0.65, "speech": 0.35, "ocr": 0}
            if mode == "video_speech"
            else {"visual": 0.55, "speech": 0.30, "ocr": 0.15}
        )
    )
    query_vector = encoder.encode_texts([query])[0]
    for result in index.search_videos(query_vector, 10, weights):
        record = result["record"]
        st.subheader(
            f"{record.video_id} · {record.start:.1f}-{record.end:.1f} c · {result['score']:.3f}"
        )
        st.video(record.path, start_time=int(record.start))
        st.caption(
            "Вклад: "
            + ", ".join(f"{key}={value:.3f}" for key, value in result["contributions"].items())
        )
        if Path(artifact_dir, record.keyframe).exists():
            st.image(str(Path(artifact_dir, record.keyframe)), width=320)
        if record.transcript:
            st.write("Речь:", record.transcript)
        if record.ocr_text:
            st.write("Текст в кадре:", record.ocr_text)
elif query:
    st.error("Сначала постройте индекс командой videoscope-index")
