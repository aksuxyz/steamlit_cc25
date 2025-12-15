import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from transformers import pipeline
import time

# Page configuration
st.set_page_config(
    page_title="Sentiment Analysis App",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sentiment-positive {
        color: #2ecc71;
        font-weight: bold;
        font-size: 2rem;
    }
    .sentiment-negative {
        color: #e74c3c;
        font-weight: bold;
        font-size: 2rem;
    }
    .sentiment-neutral {
        color: #f39c12;
        font-weight: bold;
        font-size: 2rem;
    }
    .stTextArea textarea {
        font-size: 1.1rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Cache the model to avoid reloading on every interaction
@st.cache_resource
def load_sentiment_model():
    """Load the sentiment analysis model."""
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# Initialize session state for history
if 'history' not in st.session_state:
    st.session_state.history = []

# Main title
st.markdown('<h1 class="main-header">🎭 Sentiment Analysis Application</h1>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("📊 About")
    st.write("""
    This application analyzes the sentiment of text using advanced Natural Language Processing (NLP) techniques.
    
    **Features:**
    - Real-time sentiment analysis
    - Confidence score visualization
    - Analysis history tracking
    - Batch text analysis
    """)
    
    st.header("⚙️ Settings")
    show_confidence = st.checkbox("Show confidence scores", value=True)
    show_history = st.checkbox("Show analysis history", value=True)
    
    if st.button("Clear History"):
        st.session_state.history = []
        st.success("History cleared!")

# Main content area
tab1, tab2, tab3 = st.tabs(["📝 Single Analysis", "📋 Batch Analysis", "📈 History"])

with tab1:
    st.header("Analyze Text Sentiment")
    
    # Text input
    user_input = st.text_area(
        "Enter text to analyze:",
        height=150,
        placeholder="Type or paste your text here..."
    )
    
    # Example texts
    with st.expander("💡 Try Example Texts"):
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Positive Example"):
                user_input = "I absolutely love this product! It exceeded all my expectations and made my day so much better."
        with col2:
            if st.button("Negative Example"):
                user_input = "This is the worst experience I've ever had. Completely disappointed and frustrated."
        with col3:
            if st.button("Neutral Example"):
                user_input = "The meeting is scheduled for tomorrow at 3 PM in the conference room."
    
    # Analyze button
    if st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True):
        if user_input.strip():
            with st.spinner("Analyzing sentiment..."):
                try:
                    # Load model and analyze
                    sentiment_analyzer = load_sentiment_model()
                    result = sentiment_analyzer(user_input[:512])[0]  # Limit to 512 tokens
                    
                    label = result['label']
                    score = result['score']
                    
                    # Add to history
                    st.session_state.history.append({
                        'text': user_input[:100] + "..." if len(user_input) > 100 else user_input,
                        'sentiment': label,
                        'confidence': score,
                        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
                    })
                    
                    # Display results
                    st.success("Analysis Complete!")
                    
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.subheader("Sentiment Result")
                        if label == "POSITIVE":
                            st.markdown(f'<p class="sentiment-positive">😊 {label}</p>', unsafe_allow_html=True)
                        elif label == "NEGATIVE":
                            st.markdown(f'<p class="sentiment-negative">😔 {label}</p>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<p class="sentiment-neutral">😐 {label}</p>', unsafe_allow_html=True)
                        
                        if show_confidence:
                            st.metric("Confidence Score", f"{score:.2%}")
                    
                    with col2:
                        if show_confidence:
                            fig = go.Figure(go.Indicator(
                                mode="gauge+number",
                                value=score * 100,
                                title={'text': "Confidence Level"},
                                gauge={
                                    'axis': {'range': [0, 100]},
                                    'bar': {'color': "#2ecc71" if label == "POSITIVE" else "#e74c3c"},
                                    'steps': [
                                        {'range': [0, 50], 'color': "#ecf0f1"},
                                        {'range': [50, 75], 'color': "#bdc3c7"},
                                        {'range': [75, 100], 'color': "#95a5a6"}
                                    ],
                                    'threshold': {
                                        'line': {'color': "red", 'width': 4},
                                        'thickness': 0.75,
                                        'value': 90
                                    }
                                }
                            ))
                            fig.update_layout(height=300)
                            st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"An error occurred during analysis: {str(e)}")
        else:
            st.warning("⚠️ Please enter some text to analyze.")

with tab2:
    st.header("Batch Sentiment Analysis")
    st.write("Analyze multiple texts at once. Enter one text per line.")
    
    batch_input = st.text_area(
        "Enter multiple texts (one per line):",
        height=200,
        placeholder="Text 1\nText 2\nText 3..."
    )
    
    if st.button("🔍 Analyze Batch", type="primary"):
        if batch_input.strip():
            texts = [line.strip() for line in batch_input.split('\n') if line.strip()]
            
            if texts:
                with st.spinner(f"Analyzing {len(texts)} texts..."):
                    try:
                        sentiment_analyzer = load_sentiment_model()
                        results = []
                        
                        # Progress bar
                        progress_bar = st.progress(0)
                        
                        for idx, text in enumerate(texts):
                            result = sentiment_analyzer(text[:512])[0]
                            results.append({
                                'Text': text[:50] + "..." if len(text) > 50 else text,
                                'Sentiment': result['label'],
                                'Confidence': f"{result['score']:.2%}"
                            })
                            progress_bar.progress((idx + 1) / len(texts))
                        
                        # Display results as dataframe
                        df = pd.DataFrame(results)
                        st.dataframe(df, use_container_width=True)
                        
                        # Summary statistics
                        st.subheader("📊 Summary Statistics")
                        col1, col2, col3 = st.columns(3)
                        
                        positive_count = sum(1 for r in results if r['Sentiment'] == 'POSITIVE')
                        negative_count = sum(1 for r in results if r['Sentiment'] == 'NEGATIVE')
                        
                        with col1:
                            st.metric("Total Analyzed", len(results))
                        with col2:
                            st.metric("Positive", positive_count, delta=f"{positive_count/len(results)*100:.1f}%")
                        with col3:
                            st.metric("Negative", negative_count, delta=f"{negative_count/len(results)*100:.1f}%")
                        
                        # Pie chart
                        fig = go.Figure(data=[go.Pie(
                            labels=['Positive', 'Negative'],
                            values=[positive_count, negative_count],
                            marker_colors=['#2ecc71', '#e74c3c']
                        )])
                        fig.update_layout(title="Sentiment Distribution")
                        st.plotly_chart(fig, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")
            else:
                st.warning("⚠️ Please enter at least one text to analyze.")
        else:
            st.warning("⚠️ Please enter some texts to analyze.")

with tab3:
    st.header("Analysis History")
    
    if show_history and st.session_state.history:
        # Convert history to dataframe
        history_df = pd.DataFrame(st.session_state.history)
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Analyses", len(history_df))
        with col2:
            positive_pct = (history_df['sentiment'] == 'POSITIVE').sum() / len(history_df) * 100
            st.metric("Positive %", f"{positive_pct:.1f}%")
        with col3:
            avg_confidence = history_df['confidence'].mean()
            st.metric("Avg Confidence", f"{avg_confidence:.2%}")
        
        # Display history table
        st.dataframe(
            history_df[['timestamp', 'text', 'sentiment', 'confidence']].sort_values('timestamp', ascending=False),
            use_container_width=True
        )
        
        # Sentiment over time chart
        if len(history_df) > 1:
            st.subheader("Sentiment Trend")
            sentiment_counts = history_df.groupby(['timestamp', 'sentiment']).size().reset_index(name='count')
            
            fig = go.Figure()
            for sentiment in sentiment_counts['sentiment'].unique():
                data = sentiment_counts[sentiment_counts['sentiment'] == sentiment]
                fig.add_trace(go.Scatter(
                    x=data['timestamp'],
                    y=data['count'],
                    mode='lines+markers',
                    name=sentiment,
                    line=dict(color='#2ecc71' if sentiment == 'POSITIVE' else '#e74c3c')
                ))
            
            fig.update_layout(
                title="Sentiment Analysis Over Time",
                xaxis_title="Time",
                yaxis_title="Count",
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("📭 No analysis history yet. Start analyzing texts to see your history here!")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #7f8c8d;'>
        <p>Built with ❤️ using Streamlit | Powered by DistilBERT</p>
    </div>
    """, unsafe_allow_html=True)
