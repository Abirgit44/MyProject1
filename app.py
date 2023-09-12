import streamlit as st
import numpy as np
import pandas as pd  # Import pandas for DataFrame
import plotly.express as px
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Function to load the selected dataset
def get_dataset(name_dataset):
    if name_dataset == "Iris":
        data = datasets.load_iris()
    elif name_dataset == "Breast Cancer":
        data = datasets.load_breast_cancer()
    else:
        data = datasets.load_wine()
    features = data.data
    labels = data.target
    return features, labels, data.feature_names

# Streamlit App Title
# Create a banner-like title with a 3D cube pattern background and a glowy effect
st.markdown(
    """
    <style>
        @keyframes glowing {
            0% { text-shadow: 0 0 10px rgba(246, 51, 102, 0.8); }
            50% { text-shadow: 0 0 20px rgba(246, 51, 102, 1); }
            100% { text-shadow: 0 0 10px rgba(246, 51, 102, 0.8); }
        }

        .glowing-title {
            background: url('https://www.toptal.com/designers/subtlepatterns/patterns/cube.png');
            padding: 20px;
            border-radius: 5px;
            text-align: center;
            animation: glowing 2s infinite;
        }

        .glowing-title h1 {
            color: navy;
            opacity: 1;
        }
    </style>
    <div class="glowing-title">
        <h1>🚀 Interactive SVM Classifier Explorer</h1>
    </div>
    """,
    unsafe_allow_html=True
)





st.markdown("""
        <p style="font-size: 10px;">📱 <strong>Mobile Users:</strong> Click the <strong>top left</strrong> icon to access sidebar content for instructions on using this app.</p>
        """,unsafe_allow_html=True)

# Sidebar for dataset selection
with st.sidebar:
    st.subheader("Select a Dataset :scroll:")
    dataset_name = st.selectbox("Choose a Dataset :point_down:", ("Iris", "Breast Cancer", "Wine"))

# Load the selected dataset
features, labels, feature_names = get_dataset(dataset_name)

# Define hyperparameters section
st.sidebar.subheader("🛠️ Model Hyperparameters")
st.sidebar.write("Adjust the hyperparameters to customize the SVM model.")

# Regularization Parameter (C) Slider
C = st.sidebar.slider('Regularization Parameter (C)', 0.01, 10.0, 1.0, step=0.01)


# Description of the 'C' Parameter with Custom Styling
c_description = """
<div class="custom-description">
    <span class="markdown-styled-heading">🔶 <strong>C Parameter</strong></span>
    <p>The 'C' parameter in Support Vector Machines (SVM) is a critical hyperparameter that balances margin width and classification accuracy.</p>
    <ul>
        <li>A smaller <strong>'C'</strong> promotes a wider margin but allows some misclassification, suitable for noisy or non-linearly separable data.</li>
        <li>In contrast, a larger <strong>'C'</strong> enforces a strict margin, aiming to classify all training points accurately, potentially leading to overfitting.</li>
    </ul>
    <p>'C' is chosen through techniques like cross-validation, tailored to the dataset's characteristics, to strike the right balance between model simplicity and accuracy.</p>
    <p><strong>Selected 'C' Value:</strong> {}</p>
</div>
""".format(C)

# Define custom CSS for the description
custom_css = """
<style>
.custom-description {
    background-color: rgba(0, 0, 0, 0.1);
    padding: 10px;
    border-radius: 5px;
}
.custom-description p {
    margin: 0.5rem 0;
}
.custom-description ul {
    margin: 0.5rem 0;
    padding-left: 20px;
}
</style>
"""

# Add the custom CSS and the styled description to the sidebar
st.sidebar.markdown(custom_css, unsafe_allow_html=True)
st.sidebar.markdown(c_description, unsafe_allow_html=True)




# Create the SVM classifier
clf = SVC(C=C)

# Main content
st.write("""
    ## Explore the Dataset and SVM Classifier
    Welcome to the Interactive SVM Classifier Explorer. This app allows you to analyze datasets and train SVM classifiers.
    Select a dataset from the sidebar and adjust hyperparameters as needed. Click 'Explore' to see the results.
    """)

# Display dataset information
st.write(f"**Dataset Information**")
st.write(f"- Dataset Shape: {features.shape}")
st.write(f"- Number of Classes: {len(np.unique(labels))}")

# Button to view the loaded dataset
if st.button("View Dataset"):
    st.subheader("Loaded Dataset")

    # Create a DataFrame with feature names as columns
    df = pd.DataFrame(features, columns=feature_names)

    # Display the DataFrame
    st.write(df)

    # Dictionary of dataset descriptions
    dataset_descriptions = {
        "Iris": {
            "Description": "The Iris dataset is a well-known dataset in machine learning and statistics. It contains samples of iris flowers from three different species: Setosa, Versicolor, and Virginica.",
            "Attributes": "It includes four features, namely sepal length, sepal width, petal length, and petal width, all measured in centimeters.",
            "Use Case": "The Iris dataset is commonly used for classification tasks, such as species identification based on flower characteristics."
        },
        "Breast Cancer": {
            "Description": "The Breast Cancer dataset is used for breast cancer classification tasks. It contains features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass.",
            "Attributes": "It includes various features related to the cell nuclei's characteristics, such as radius, texture, perimeter, area, smoothness, and more.",
            "Use Case": "This dataset is often used for binary classification tasks, where the goal is to distinguish between benign and malignant breast tumors."
        },
        "Wine": {
            "Description": "The Wine dataset consists of data from three different types of wine (classes) grown in the same region of Italy. It includes measurements of various chemical components in each wine.",
            "Attributes": "The dataset includes 13 features, such as alcohol content, malic acid, ash, alkalinity of ash, magnesium, and more, all measured in different units.",
            "Use Case": "The Wine dataset is often used for classification tasks, where the goal is to classify wines into one of the three classes based on their chemical properties."
        }
    }
    def display_dataset_description(dataset_name):
        description = dataset_descriptions[dataset_name]
        st.markdown(
            f'<div class="description-block">'
            f'<h3 style="font-weight: bold; color: white; text-shadow: 0 0 10px blue;">Description:</h3>'
            f'<p style="font-size: 14px; line-height: 1.4; color: white; text-shadow: 0 0 10px purple;">{description["Description"]}</p>'
            f'</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            f'<div class="description-block">'
            f'<h3 style="font-weight: bold; color: white; text-shadow: 0 0 10px blue;">Attributes:</h3>'
            f'<p style="font-size: 14px; line-height: 1.4; color: white; text-shadow: 0 0 10px purple;">{description["Attributes"]}</p>'
            f'</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            f'<div class="description-block">'
            f'<h3 style="font-weight: bold; color: white; text-shadow: 0 0 10px blue;">Use Case:</h3>'
            f'<p style="font-size: 14px; line-height: 1.4; color: white; text-shadow: 0 0 10px purple;">{description["Use Case"]}</p>'
            f'</div>',
            unsafe_allow_html=True
        )


    if dataset_name in dataset_descriptions:
            display_dataset_description(dataset_name)

# Train-test split
a_train, a_test, b_train, b_test = train_test_split(features, labels, test_size=0.2, random_state=1234)

# Train the SVM classifier
if st.button('Explore'):
    st.subheader("Model Training and Evaluation")

    # Fit the classifier
    clf.fit(a_train, b_train)

    # Make predictions
    b_pred = clf.predict(a_test)

    # Calculate accuracy
    accuracy = accuracy_score(b_test, b_pred)

    # Display accuracy
    st.write(f"**Accuracy**: {accuracy:.2%}")

    # Visualize PCA in 3D
    st.subheader("Principal Component Analysis (PCA) - Interactive 3D Visualization")
    st.write("Visualizing the dataset in 3D using PCA")

    # Perform PCA for 3D visualization
    pca = PCA(3)
    a_projected = pca.fit_transform(features)

    # Map dataset names to more understandable class labels
    class_labels = {
        "Iris": "Iris Species",
        "Breast Cancer": "Diagnosis (Malignant/Benign)",
        "Wine": "Wine Types"
    }

    # Create interactive 3D scatter plot using Plotly with labeled classes
    fig = px.scatter_3d(
        a_projected, x=a_projected[:, 0], y=a_projected[:, 1], z=a_projected[:, 2],
        color=labels, opacity=0.7, title=f"PCA 3D Visualization - {class_labels[dataset_name]}"
    )


    # Customize axes labels
    fig.update_layout(scene=dict(xaxis_title='Principal Component 1', yaxis_title='Principal Component 2', zaxis_title='Principal Component 3'))

    st.plotly_chart(fig)

# Provide information about SVM
st.write("""
    ## About Support Vector Machine (SVM)
    Support Vector Machine is a powerful classification algorithm that aims to find the best separating hyperplane.
    The 'C' parameter controls the trade-off between maximizing margin and minimizing errors.
    Adjust 'C' using the sidebar slider to observe its impact on classifier performance.
    """)

# Add a footer or additional information
st.markdown("""
    ## About
    This Streamlit app was created for exploring SVM classifiers using various datasets.
    Feel free to experiment with different datasets and hyperparameters to understand SVM behavior.
    For more information and code, check out the: [![https://img.shields.io/badge/GitHub-Repository-blue?logo=github](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/Abirgit44/PCA-analysis-using-Streamlit)
    """)
