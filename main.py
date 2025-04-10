import os
import yaml
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime
import argparse

# Model imports
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from hdbscan import HDBSCAN

# Gensim imports for coherence calculation
import gensim
import gensim.corpora as corpora
from gensim.models.coherencemodel import CoherenceModel

# Dashboard imports
import dash
from dash import dcc, html, dash_table
import plotly.express as px
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("topic_modeling.log")
    ]
)
logger = logging.getLogger(__name__)


class DataLoader:
    """Load and preprocess the data for topic modeling."""
    
    def __init__(self, config: Dict):
        """
        Initialize the DataLoader.
        
        Args:
            config: Dictionary containing data configuration parameters
        """
        self.input_filepath = config["input_filepath"]
        self.review_column = config.get("review_column", "review")
        self.sample_size = config.get("sample_size", None)
        
    def load_reviews(self) -> pd.Series:
        """
        Load and preprocess reviews from a CSV file.
        
        Returns:
            pd.Series: Series of review texts with missing values removed
        """
        logger.info(f"Loading data from {self.input_filepath}")
        try:
            df = pd.read_csv(self.input_filepath)
            logger.info(f"Total documents: {len(df)}")
            
            # Check if review column exists
            if self.review_column not in df.columns:
                available_cols = ", ".join(df.columns)
                logger.error(f"Review column '{self.review_column}' not found. Available columns: {available_cols}")
                raise ValueError(f"Review column '{self.review_column}' not found in dataset")
            
            # Report missing values
            missing = df[self.review_column].isna().sum()
            logger.info(f"Missing values: {missing} ({missing/len(df):.2%})")
            
            # Remove missing values
            reviews = df[self.review_column].dropna()
            logger.info(f"Usable documents: {len(reviews)}")
            
            # Sample data if requested
            if self.sample_size is not None and self.sample_size < len(reviews):
                logger.info(f"Sampling {self.sample_size} documents")
                reviews = reviews.sample(self.sample_size, random_state=42)
            
            # Basic text cleaning
            reviews = reviews.apply(lambda x: str(x).strip())
            
            # Filter out empty strings after cleaning
            reviews = reviews[reviews.str.len() > 0]
            logger.info(f"Final document count: {len(reviews)}")
            
            return reviews
            
        except FileNotFoundError:
            logger.error(f"File not found: {self.input_filepath}")
            raise
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise


class BERTopicConfigurator:
    """Configure the BERTopic model with the provided parameters."""
    
    def __init__(self, config: Dict):
        """
        Initialize the BERTopic model configuration.
        
        Args:
            config: Dictionary containing model configuration parameters
        """
        logger.info("Configuring BERTopic model")
        self.config = config
        
        # Initialize embedding model
        try:
            self.embedding_model = SentenceTransformer(self.config["transformer_name"])
            logger.info(f"Initialized embedding model: {self.config['transformer_name']}")
        except Exception as e:
            logger.error(f"Error initializing embedding model: {str(e)}")
            raise
        
        # Initialize UMAP for dimensionality reduction
        self.umap_model = UMAP(**self.config["umap"])
        logger.info(f"Initialized UMAP with parameters: {self.config['umap']}")
        
        # Initialize HDBSCAN for clustering
        self.hdbscan_model = HDBSCAN(**self.config["hdbscan"])
        logger.info(f"Initialized HDBSCAN with parameters: {self.config['hdbscan']}")
        
        # Initialize CountVectorizer for feature extraction
        self.vectorizer_model = CountVectorizer(
            stop_words=None,  # self.config["vectorizer"]["stop_words"],
            ngram_range=tuple(self.config["vectorizer"]["ngram_range"])
        )
        logger.info(f"Initialized CountVectorizer with parameters: {self.config['vectorizer']}")
        
        # Initialize BERTopic model
        self.topic_model = BERTopic(
            min_topic_size=self.config["min_topic_size"],
            embedding_model=self.embedding_model,
            umap_model=self.umap_model,
            hdbscan_model=self.hdbscan_model,
            vectorizer_model=self.vectorizer_model,
            language=self.config["language"],
            nr_topics=self.config["nr_topics"],
            verbose=True
        )
        logger.info("BERTopic model configuration complete")


class TopicModeler:
    """Train and manage the topic model."""
    
    def __init__(self, topic_model: BERTopic):
        """
        Initialize the topic modeler.
        
        Args:
            topic_model: Configured BERTopic model
        """
        self.topic_model = topic_model
        self.topics = None
        self.probs = None
        self.doc_info = None
        
    def fit(self, documents: List[str]) -> Tuple[List[int], np.ndarray]:
        """
        Train the model on documents and return topics and probabilities.
        
        Args:
            documents: List of document texts
            
        Returns:
            Tuple containing topic assignments and topic probabilities
        """
        logger.info("Training BERTopic model")
        
        try:
            self.topics, self.probs = self.topic_model.fit_transform(documents)
            info = self.topic_model.get_topic_info()
            logger.info(f"Number of topics identified: {len(info)}")
            logger.info(f"Distribution: {info['Count'].value_counts().to_dict()}")
            #check
            logger.info(f"Topic assignments: {self.topics[:3]}")
            logger.info(f"Topic probabilities: {self.probs[:3]}")

            # Create document info dataframe
            self.doc_info = pd.DataFrame({
                'Document': documents,
                'Topic': self.topics,
                'Probability': self.probs
            })
            
            return self.topics, self.probs
        except Exception as e:
            logger.error(f"Error during model training: {str(e)}")
            raise
    
    def save(self, output_dir: str):
        """
        Save the BERTopic model and related outputs.
        
        Args:
            output_dir: Directory to save model artifacts
        """
        # Create timestamped directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(output_dir, f"bertopic_model_{timestamp}")
        os.makedirs(save_path, exist_ok=True)
        
        logger.info(f"Saving model to {save_path}")
        
        try:
            # Save the BERTopic model
            self.topic_model.save(os.path.join(save_path, "model"), serialization="safetensors")
            
            # Save document info
            if self.doc_info is not None:
                self.doc_info.to_csv(os.path.join(save_path, "document_topics.csv"), index=False)
            
            # Save topic info
            topic_info = self.topic_model.get_topic_info()
            topic_info.to_csv(os.path.join(save_path, "topic_info.csv"), index=False)
            
            # Save top terms per topic
            topic_terms = {}
            for topic in topic_info['Topic'].unique():
                if topic != -1:  # Skip outlier topic
                    terms = self.topic_model.get_topic(topic)
                    topic_terms[topic] = terms
            
            pd.DataFrame(topic_terms).to_csv(os.path.join(save_path, "topic_terms.csv"))
            
            logger.info(f"Model and artifacts saved successfully to {save_path}")
            return save_path
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise


class Evaluator:
    """Evaluate the topic model using various metrics."""
    
    def __init__(self, topic_model: BERTopic, config: Dict):
        """
        Initialize the evaluator.
        
        Args:
            topic_model: Trained BERTopic model
            config: Dictionary containing evaluation parameters
        """
        self.topic_model = topic_model
        self.config = config
        self.sample_size = config.get("sample_size", 1000)
        self.coherence_metrics = config.get("coherence_metrics", ["c_v"])
        
    def calculate_coherence(self, documents: List[str]) -> Dict[str, float]:
        """
        Calculate coherence scores for the topic model using Gensim's CoherenceModel.
        
        Args:
            documents: List of document texts
            
        Returns:
            Dictionary mapping coherence metrics to their scores
        """
        logger.info("Calculating coherence scores using Gensim")
        scores = {}
        
        # Sample documents if necessary for performance
        if self.sample_size and len(documents) > self.sample_size:
            logger.info(f"Sampling {self.sample_size} documents for coherence calculation")
            sampled_docs = np.random.choice(documents, self.sample_size, replace=False).tolist()
        else:
            sampled_docs = documents
        
        try:
            # Tokenize documents for Gensim
            logger.info("Tokenizing documents for coherence calculation")
            tokenized_texts = [doc.split() for doc in sampled_docs if doc]
            
            # Extract topic words from BERTopic model
            logger.info("Extracting topic words from model")
            topics_dict = self.topic_model.get_topics()
            
            # Skip outlier topic (-1) if present
            if -1 in topics_dict:
                del topics_dict[-1]
                
            # Format topics for Gensim coherence model
            gensim_topics = []
            for topic_id in sorted(topics_dict.keys()):
                # Extract words without weights
                topic_words = [word for word, _ in topics_dict[topic_id]]
                gensim_topics.append(topic_words)
            
            # Create Gensim dictionary
            logger.info("Creating Gensim dictionary")
            dictionary = corpora.Dictionary(tokenized_texts)
            
            # Calculate coherence for each metric
            for metric in self.coherence_metrics:
                try:
                    logger.info(f"Calculating {metric} coherence")
                    cm = CoherenceModel(
                        topics=gensim_topics,
                        texts=tokenized_texts,
                        dictionary=dictionary,
                        coherence=metric
                    )
                    score = cm.get_coherence()
                    scores[metric] = score
                    logger.info(f"{metric} coherence: {score:.4f}")
                except Exception as e:
                    logger.error(f"Error calculating {metric} coherence: {str(e)}")
                    scores[metric] = None
            
            logger.info(f"Coherence scores: {scores}")
            return scores
        except Exception as e:
            logger.error(f"Error in coherence calculation pipeline: {str(e)}")
            return scores
    
    def calculate_topic_diversity(self) -> float:
        """
        Calculate topic diversity score.
        
        Returns:
            Topic diversity score (0-1)
        """
        try:
            # Get all topics except for outliers
            topics = self.topic_model.get_topics()
            if -1 in topics:
                del topics[-1]
            
            if not topics:
                logger.warning("No topics found for diversity calculation")
                return 0.0
            
            # Get unique words across all topics
            unique_words = set()
            for topic in topics.values():
                unique_words.update([word for word, _ in topic])
            
            # Calculate diversity as ratio of unique words to total words
            total_words = sum(len(topic) for topic in topics.values())
            diversity = len(unique_words) / total_words if total_words > 0 else 0
            
            logger.info(f"Topic diversity: {diversity:.4f}")
            return diversity
        except Exception as e:
            logger.error(f"Error calculating topic diversity: {str(e)}")
            return 0.0


def visualize_topics_2d(topics, reduced_embeddings, docs, width=1000, height=700):
    """
    Visualize topics in 2D using Plotly.
    
    Args:
        topics: List of topic assignments
        reduced_embeddings: 2D embeddings for visualization
        docs: List of document texts
        width: Plot width
        height: Plot height
        
    Returns:
        Plotly Figure object
    """
    if reduced_embeddings.shape[1] < 2:
        logger.error("Reduced embeddings must have at least two dimensions")
        raise ValueError("Reduced embeddings must have at least two dimensions")
    
    # Create dataframe for plotting
    df = pd.DataFrame({
        'x': reduced_embeddings[:, 0],
        'y': reduced_embeddings[:, 1],
        'Topic': [f"Topic {t}" if t != -1 else "Outliers (-1)" for t in topics],
        'Review': [doc[:100] + "..." if len(doc) > 100 else doc for doc in docs]  # Truncate long texts
    })
    
    # Create scatter plot
    fig = px.scatter(
        df,
        x='x',
        y='y',
        color='Topic',
        hover_data={'Review': True, 'Topic': True, 'x': False, 'y': False},
        title="Topic Visualization in 2D Space",
        labels={'x': 'Dimension 1', 'y': 'Dimension 2'},
        width=width,
        height=height
    )
    
    # Customize appearance
    fig.update_traces(marker=dict(size=8, opacity=0.7))
    fig.update_layout(
        legend_title_text='Topics',
        showlegend=True,
        title_x=0.5,
        hovermode='closest',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    return fig


def build_dashboard(topic_model, topics, probs, docs, reduced_embeddings, coherence_scores, 
                    topic_diversity, port: int):
    """
    Build an interactive Dash dashboard to visualize BERTopic results.
    
    Args:
        topic_model: Trained BERTopic model
        topics: List of topic assignments
        probs: Probabilities associated with topic assignments
        docs: Original document texts
        reduced_embeddings: 2D reduced embeddings for visualization
        coherence_scores: Dictionary of coherence scores
        topic_diversity: Topic diversity score
        port: Port number for the Dash server
        
    Returns:
        Dash application instance
    """
    logger.info("Creating dashboard")
    app = dash.Dash(
        __name__, 
        external_stylesheets=[
            "https://cdnjs.cloudflare.com/ajax/libs/tailwindcss/2.2.19/tailwind.min.css"
        ]
    )
    
    # Topic information
    topic_info = topic_model.get_topic_info()
    
    # Get some sample documents for each topic
    topic_docs = {}
    for topic_id in set(topics):
        if topic_id == -1:  # Skip outliers for document samples
            continue
        indices = [i for i, t in enumerate(topics) if t == topic_id]
        if indices:
            # Get up to 5 sample documents for each topic

            sample_indices = indices[:5]
            topic_docs[topic_id] = [(docs[i], probs[i]) for i in sample_indices]
 
    # Define dashboard layout
    app.layout = html.Div([
        # Header
        html.Div([
            html.H1("BERTopic Dashboard", className="text-2xl font-bold mb-2"),
            html.P("Interactive visualization of topic modeling results", className="text-gray-600"),
        ], className="p-4 bg-gray-100 border-b"),
        
        # Main content
        html.Div([
            # Metrics and Overview
            html.Div([
                html.H2("Model Performance", className="text-xl font-bold mb-3"),
                html.Div([
                    # Coherence scores
                    html.Div([
                        html.H3("Coherence Scores", className="text-lg font-semibold mb-2"),
                        dash_table.DataTable(
                            id="coherence-table",
                            columns=[
                                {"name": "Metric", "id": "Metric"},
                                {"name": "Score", "id": "Score"}
                            ],
                            data=[{"Metric": metric, "Score": f"{score:.4f}" if score is not None else "N/A"} 
                                 for metric, score in coherence_scores.items()],
                            style_table={"overflowX": "auto"},
                            style_cell={"textAlign": "left", "padding": "10px"},
                            style_header={"backgroundColor": "rgb(230, 230, 230)", "fontWeight": "bold"}
                        ),
                    ], className="p-4 border rounded bg-white mb-4"),
                    
                    # Topic diversity
                    html.Div([
                        html.H3("Topic Diversity", className="text-lg font-semibold mb-2"),
                        html.Div([
                            html.Span(f"{topic_diversity:.4f}", className="text-2xl font-bold"),
                            html.Span(" / 1.0", className="text-gray-500")
                        ], className="flex items-center"),
                        html.P("Higher scores indicate more diverse topics", className="text-sm text-gray-600 mt-1")
                    ], className="p-4 border rounded bg-white mb-4"),
                
                    # Topic summary
                    
                    html.Div([
                        html.H3("Topic Summary", className="text-lg font-semibold mb-2"),
                        dash_table.DataTable(
                            id="topic-summary",
                            columns=[
                                {"name": "Topic", "id": "Topic"},
                                {"name": "Count", "id": "Count"},
                                {"name": "Name", "id": "Name"},
                            ],

                            # data=topic_info.to_dict("records"),
                            data=topic_info[["Topic", "Count", "Name"]].to_dict("records"),
                            style_table={"overflowX": "auto"},
                            style_cell={"textAlign": "left", "padding": "10px"},
                            style_header={"backgroundColor": "rgb(230, 230, 230)", "fontWeight": "bold"},
                            page_size=10
                        ),
                    ], className="p-4 border rounded bg-white"),
                ], className="mb-6"),
            ], className="w-full lg:w-1/3 p-4"),
            

            # Visualizations
            html.Div([
                # Topic visualization tabs
                html.Div([
                    html.H2("Topic Visualizations", className="text-xl font-bold mb-3"),
                    dcc.Tabs([
                        dcc.Tab(label="Topic Overview", children=[
                            dcc.Graph(
                                id="topics-overview",
                                figure=topic_model.visualize_topics(),
                                style={"height": "600px"}
                            )
                        ], className="p-4"),

                        dcc.Tab(label="Document Map", children=[
                            dcc.Graph(
                                id="documents-map",
                                figure=visualize_topics_2d(
                                    topics=topics,
                                    reduced_embeddings=reduced_embeddings,
                                    docs=docs,
                                    width=800,
                                    height=800
                                ),
                                style={"height": "600px"}
                            )
                        ], className="p-4"),
                        dcc.Tab(label="Topic Hierarchy", children=[
                            dcc.Graph(
                                id="hierarchy-graph",
                                figure=topic_model.visualize_hierarchy(),
                                style={"height": "600px"}
                            )
                        ], className="p-4"),
                        dcc.Tab(label="Topic Terms", children=[
                            dcc.Graph(
                                id="barchart-graph",
                                figure=topic_model.visualize_barchart(top_n_topics=10),
                                style={"height": "600px"}
                            )
                        ], className="p-4"),
                        dcc.Tab(label="Topic Distribution", children=[
                            dcc.Graph(
                                id="distribution-graph",
                                figure=topic_model.visualize_distribution(probs[:10], topics[:10]),
                                style={"height": "600px"}
                            )
                        ], className="p-4"),
                    ], className="mb-4"),
                ], className="mb-6 bg-white border rounded p-4"),
                
                

                # Topic explorer
                html.Div([
                    html.H2("Topic Explorer", className="text-xl font-bold mb-3"),
                    html.Div([
                        html.Label("Select Topic:", className="block mb-2 font-semibold"),
                        dcc.Dropdown(
                            id="topic-selector",
                            options=[
                                {"label": f"Topic {row['Topic']}: {row['Name']}", "value": row["Topic"]}
                                for _, row in topic_info.iterrows() if row["Topic"] != -1
                            ],
                            value=topic_info.iloc[1]["Topic"] if len(topic_info) > 1 else None,
                            className="mb-4"
                        ),
                        html.Div(id="topic-details", className="mt-4"),
                    ], className="p-4"),
                ], className="bg-white border rounded p-4"),
            ], className="w-full lg:w-2/3 p-4"),
        ], className="flex flex-wrap"),
    ], className="container mx-auto pb-10")
    
    # Topic details callback
    @app.callback(
        Output("topic-details", "children"),
        Input("topic-selector", "value")
    )
    def update_topic_details(topic_id):
        if topic_id is None:
            return html.Div("Select a topic to view details", className="text-gray-500")
        
        # Get topic terms
        terms = topic_model.get_topic(topic_id)
        
        # Create topic word table
        terms_table = dash_table.DataTable(
            columns=[
                {"name": "Term", "id": "term"},
                {"name": "Weight", "id": "weight"}
            ],
            data=[{"term": term, "weight": f"{weight:.4f}"} for term, weight in terms],
            style_table={"overflowX": "auto"},
            style_cell={"textAlign": "left", "padding": "10px"},
            style_header={"backgroundColor": "rgb(230, 230, 230)", "fontWeight": "bold"}
        )
        
        # Create sample documents section
        samples_div = html.Div([
            html.H4("Sample Documents", className="text-lg font-semibold mb-2"),
            html.P("No sample documents available", className="text-gray-500")
        ])
        
        if topic_id in topic_docs:
            sample_items = []
            for doc, prob in topic_docs[topic_id]:
                sample_items.append(html.Div([
                    html.P(f"{doc[:200]}..." if len(doc) > 200 else doc, className="mb-1"),
                    html.P(f"Probability: {prob:.4f}", className="text-sm text-gray-600 mb-3")
                ]))
            
            samples_div = html.Div([
                html.H4("Sample Documents", className="text-lg font-semibold mb-2"),
                html.Div(sample_items, className="border-t pt-2")
            ])
        
        return html.Div([
            html.H3(f"Topic {topic_id}: {topic_model.get_topic_info().loc[topic_model.get_topic_info()['Topic'] == topic_id, 'Name'].values[0]}", 
                   className="text-xl font-semibold mb-3"),
            html.Div([
                html.H4("Top Terms", className="text-lg font-semibold mb-2"),
                terms_table,
            ], className="mb-4"),
            samples_div
        ])
    
    logger.info(f"Dashboard created, ready to run on port {port}")
    return app


def main(config_path: str):
    """
    Main function to run the topic modeling pipeline.
    
    Args:
        config_path: Path to the configuration YAML file
    """
    try:
        # Load configuration
        logger.info(f"Loading configuration from {config_path}")
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        
        # Create output directory if it doesn't exist
        output_dir = config["output"]["output_dir"]
        os.makedirs(output_dir, exist_ok=True)
        
        # Load data
        data_loader = DataLoader(config["data"])
        reviews = data_loader.load_reviews()
        
        if len(reviews) == 0:
            logger.error("No usable data available. Stopping pipeline.")
            return
        
        # Configure and train model
        configurer = BERTopicConfigurator(config["model"])
        modeler = TopicModeler(configurer.topic_model)
        topics, probs = modeler.fit(reviews.tolist())
        
        # Evaluate model
        evaluator = Evaluator(modeler.topic_model, config["evaluation"])
        coherence_scores = evaluator.calculate_coherence(reviews.tolist())
        topic_diversity = evaluator.calculate_topic_diversity()
        
        # Save model if configured
        if config["output"]["save_model"]:
            model_path = modeler.save(output_dir)
            logger.info(f"Model saved to {model_path}")
        
        # Prepare visualization data
        logger.info("Computing embeddings for visualization")
        embeddings = configurer.embedding_model.encode(reviews.tolist(), show_progress_bar=True)
        
        # Reduce dimensionality to 2D for visualization
        viz_umap = UMAP(n_components=2, min_dist=0.0, metric='cosine')
        reduced_embeddings = viz_umap.fit_transform(embeddings)
        
        # Create and run dashboard
        app = build_dashboard(
            topic_model=modeler.topic_model,
            topics=topics,
            probs=probs,
            docs=reviews.tolist(),
            reduced_embeddings=reduced_embeddings,
            coherence_scores=coherence_scores,
            topic_diversity=topic_diversity,
            port=config["output"]["dashboard_port"]
        )
        
        logger.info(f"Starting dashboard on port {config['output']['dashboard_port']}")
        app.run(host="0.0.0.0", port=config["output"]["dashboard_port"], debug=True)
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {str(e)}")
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML configuration: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        logger.debug("Error details:", exc_info=True)


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="BERTopic modeling pipeline")
    parser.add_argument(
        "--config", 
        type=str, 
        default="config.yaml",
        help="Path to configuration YAML file"
    )
    args = parser.parse_args()
    
    # Set environment variable to avoid tokenizers parallelism issues
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # Run the pipeline
    main(args.config)