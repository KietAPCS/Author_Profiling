import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'source')))

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime
import time

from preprocess import stem_lema
from count_word import count_word
from similarity import calculate_similarity

import nltk
nltk.download('punkt_tab', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

# Page configuration
st.set_page_config(
    page_title="✍️ Author Profiling Analytics",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .main-header h1 {
        color: white;
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .main-header p {
        color: #f8f9fa;
        font-size: 1.2rem;
        margin: 0;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    .comparison-result {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
    }
    .stTextArea > div > div > textarea {
        background: #f8f9fa;
        border-radius: 10px;
        border: 2px solid #e9ecef !important;
        transition: border-color 0.3s;
    }
    .stTextArea > div > div > textarea:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 0.2rem rgba(102, 126, 234, 0.25) !important;
    }
    .sidebar-info {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 1rem;
    }
    .stProgress .st-bo {
        background-color: #667eea;
    }
    .analyze-button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        font-weight: 600;
        font-size: 1.1rem;
        cursor: pointer;
        transition: all 0.3s;
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

THRESHOLD = 0.8

def get_similarity_interpretation(similarity):
    """Get interpretation of similarity score"""
    if similarity >= 0.9:
        return "Extremely High", "🟢", "Very likely same author"
    elif similarity >= 0.75:
        return "High", "🟡", "Likely same author"
    elif similarity >= 0.5:
        return "Moderate", "🟠", "Possibly same author"
    elif similarity >= 0.25:
        return "Low", "🔴", "Unlikely same author"
    else:
        return "Very Low", "⚫", "Very unlikely same author"

def create_similarity_gauge(similarity):
    """Create a beautiful gauge chart for similarity"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = similarity * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Similarity Score", 'font': {'size': 24, 'color': '#2c3e50'}},
        delta = {'reference': THRESHOLD * 100, 'position': "top"},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 25], 'color': 'lightgray'},
                {'range': [25, 50], 'color': 'yellow'},
                {'range': [50, 75], 'color': 'orange'},
                {'range': [75, 100], 'color': 'lightgreen'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': THRESHOLD * 100
            }
        }
    ))
    
    fig.update_layout(
        height=400,
        font={'color': "darkblue", 'family': "Arial"},
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def analyze_text_stats(text):
    """Analyze basic text statistics"""
    if not text.strip():
        return {}
    
    tokens = stem_lema(text)
    sentences = text.split('.')
    words = text.split()
    
    return {
        'word_count': len(words),
        'sentence_count': len([s for s in sentences if s.strip()]),
        'character_count': len(text),
        'processed_tokens': len(tokens),
        'avg_sentence_length': len(words) / max(len([s for s in sentences if s.strip()]), 1),
        'avg_word_length': sum(len(word) for word in words) / max(len(words), 1)
    }

def compare_authors(original_text, test_text, ngram_size=2):
    """Compare authorship between two texts"""
    if not original_text.strip() or not test_text.strip():
        return None
    
    # Check minimum text length
    if len(original_text.split()) < ngram_size or len(test_text.split()) < ngram_size:
        return 0.0
    
    try:
        original_tokens = stem_lema(original_text)
        test_tokens = stem_lema(test_text)
        
        # Check if we have enough tokens after preprocessing
        if len(original_tokens) < ngram_size or len(test_tokens) < ngram_size:
            return 0.0

        original_profile = count_word(original_tokens, n=ngram_size)
        test_profile = count_word(test_tokens, n=ngram_size)
        
        similarity = calculate_similarity(original_profile, test_profile)
        return similarity
    
    except Exception as e:
        print(f"Error in compare_authors: {e}")
        return None

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>✍️ Author Profiling Analytics</h1>
        <p>Advanced text analysis for authorship verification using NLP techniques</p>
        <text> - KietAPCS - </text>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-info">
            <h3>📊 How It Works</h3>
            <p>This tool analyzes writing style using:</p>
            <ul>
                <li>N-gram analysis</li>
                <li>Text preprocessing</li>
                <li>Cosine similarity</li>
                <li>Statistical features</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### ⚙️ Settings")
        ngram_size = st.slider("N-gram Size", 1, 5, 2, help="Size of n-grams for analysis")
        threshold = st.slider("Similarity Threshold", 0.0, 1.0, THRESHOLD, 0.05, 
                             help="Threshold for determining authorship match")
        
        st.markdown("### 📈 Analysis Features")
        show_stats = st.checkbox("Show Text Statistics", True)
        show_gauge = st.checkbox("Show Similarity Gauge", True)
        show_comparison = st.checkbox("Show Detailed Comparison", True)

    # Main content
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📝 Original Text")
        original_text = st.text_area(
            "Enter the original text for comparison:",
            height=300,
            placeholder="Paste the original author's text here...",
            help="Enter a substantial text sample for better accuracy"
        )
        
        if show_stats and original_text:
            orig_stats = analyze_text_stats(original_text)
            if orig_stats:
                st.markdown("#### 📊 Original Text Statistics")
                col1a, col1b, col1c = st.columns(3)
                with col1a:
                    st.metric("Words", orig_stats['word_count'])
                with col1b:
                    st.metric("Sentences", orig_stats['sentence_count'])
                with col1c:
                    st.metric("Characters", orig_stats['character_count'])

    with col2:
        st.markdown("### 🔍 Test Text")
        test_text = st.text_area(
            "Enter the test text for comparison:",
            height=300,
            placeholder="Paste the text to analyze here...",
            help="Enter the text you want to verify authorship for"
        )
        
        if show_stats and test_text:
            test_stats = analyze_text_stats(test_text)
            if test_stats:
                st.markdown("#### 📊 Test Text Statistics")
                col2a, col2b, col2c = st.columns(3)
                with col2a:
                    st.metric("Words", test_stats['word_count'])
                with col2b:
                    st.metric("Sentences", test_stats['sentence_count'])
                with col2c:
                    st.metric("Characters", test_stats['character_count'])

    # Analysis button
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔬 Analyze Authorship", type="primary", use_container_width=True):
        if not original_text.strip() or not test_text.strip():
            st.error("❌ Please enter both original and test texts for comparison.")
            return
        
        # Check minimum text requirements
        min_words = max(10, ngram_size * 3)  # At least 10 words or 3x the n-gram size
        orig_word_count = len(original_text.split())
        test_word_count = len(test_text.split())
        
        if orig_word_count < min_words or test_word_count < min_words:
            if orig_word_count < ngram_size or test_word_count < ngram_size:
                st.error(f"❌ Texts must contain at least {ngram_size} words each for {ngram_size}-gram analysis.")
                return
        
        # Show progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("Preprocessing texts...")
        progress_bar.progress(25)
        time.sleep(0.5)
        
        status_text.text("Generating profiles...")
        progress_bar.progress(50)
        time.sleep(0.5)
        
        status_text.text("Calculating similarity...")
        progress_bar.progress(75)
        
        similarity = compare_authors(original_text, test_text, ngram_size)
        
        progress_bar.progress(100)
        status_text.text("Analysis complete!")
        time.sleep(0.5)
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        if similarity is not None:
            # Results section
            st.markdown("## 📊 Analysis Results")
            
            interpretation, emoji, description = get_similarity_interpretation(similarity)
            
            # Main result display
            st.markdown(f"""
            <div class="comparison-result">
                <h2>{emoji} Similarity Score: {similarity:.3f}</h2>
                <h3>Interpretation: {interpretation}</h3>
                <p style="font-size: 1.2rem; margin-top: 1rem;">{description}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Detailed results
            col3, col4 = st.columns([1, 1])
            
            with col3:
                if show_gauge:
                    st.plotly_chart(create_similarity_gauge(similarity), use_container_width=True)
            
            with col4:
                st.markdown("### 🎯 Key Metrics")
                
                # Confidence level
                confidence = "High" if similarity > threshold else "Low"
                confidence_color = "🟢" if similarity > threshold else "🔴"
                
                st.markdown(f"""
                **Similarity Score:** {similarity:.3f}  
                **Threshold:** {threshold:.3f}  
                **Confidence:** {confidence_color} {confidence}  
                **Match Status:** {'✅ Likely Match' if similarity > threshold else '❌ No Match'}  
                **Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}
                """)
                
                # Additional insights
                if show_comparison and original_text and test_text:
                    orig_stats = analyze_text_stats(original_text)
                    test_stats = analyze_text_stats(test_text)
                    
                    if orig_stats and test_stats:
                        st.markdown("### 📈 Text Comparison")
                        
                        comparison_data = {
                            'Metric': ['Words', 'Sentences', 'Avg Sentence Length', 'Avg Word Length'],
                            'Original': [
                                orig_stats['word_count'],
                                orig_stats['sentence_count'],
                                f"{orig_stats['avg_sentence_length']:.1f}",
                                f"{orig_stats['avg_word_length']:.1f}"
                            ],
                            'Test': [
                                test_stats['word_count'],
                                test_stats['sentence_count'],
                                f"{test_stats['avg_sentence_length']:.1f}",
                                f"{test_stats['avg_word_length']:.1f}"
                            ]
                        }
                        
                        df = pd.DataFrame(comparison_data)
                        st.dataframe(df, use_container_width=True, hide_index=True)
            
            # Recommendations
            st.markdown("### 💡 Recommendations")
            if similarity >= threshold:
                st.success("✅ Strong similarity detected. Texts likely share the same authorship.")
            else:
                st.info("ℹ️ Low similarity. Texts likely have different authors or require more data.")
        
        else:
            st.error("❌ Analysis failed. Please check your input texts.")

if __name__ == "__main__":
    main()
