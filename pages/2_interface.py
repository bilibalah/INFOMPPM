import streamlit as st
import graphviz


st.title("How personalized do you want your content?")

st.write("Use the slider below to select your preferred content style. The more personalized, the more tailored the content will be to your interests, while a more diverse selection will include a wider range of topics.")


# Define the labels for the slider
labels = ["More Personalized", "Slightly Personalized", "Balanced", "Slightly Diverse", "More Diverse"]

# Create the select slider
label = st.select_slider("Content Style", options=labels, value="Balanced")

# Map to 0.0–1.0
relevance_serendipity = labels.index(label) / (len(labels) - 1)

st.write(f"Value: {relevance_serendipity:.1f}")

st.subheader("How do the recommendations work?")

st.write("The recommendation process involves several steps to ensure that the content you receive is tailored to your preferences while maintaining a balance between relevance and diversity. Here's a breakdown of the process:")


# Create a flowchart to illustrate the recommendation process
graph = graphviz.Digraph()
graph.attr(rankdir='LR')

graph.node('A', 'User input\nPreferences & slider', shape='box', style='filled', fillcolor='#EEEDFE')
graph.node('B', 'Candidate retrieval\nPull from corpus', shape='box', style='filled', fillcolor='#E1F5EE')
graph.node('C', 'Filtering & ranking\nScore, filter, sort', shape='box', style='filled', fillcolor='#FAEEDA')
graph.node('D', 'Personalization\nAdjust for user', shape='box', style='filled', fillcolor='#FAECE7')

graph.edge('A', 'B')
graph.edge('B', 'C')
graph.edge('C', 'D')

st.graphviz_chart(graph)