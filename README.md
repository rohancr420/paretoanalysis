# Implementing Pareto Principle using ML
This is a project to analyse and figure out how to get 80% of results with 20% of effort as stated by the Pareto Principle.
It is implemented using Machine Learning libraries.
The input data is a pdf file. 
The script deals with the following:
- Extracts text data from pdf
- Extracts topics from this text data
- uses Bidirectional encoder to figure out the importance within the context
- uses transformers and MiniLM to understand importance of extracted keywords
- Uses HDBSCAN clutering algorithm to map this information in N dimensional space
- The centroid distance of these clusters are calculated to extract the pareto score by sorting data in descending order of importance
- The frequency count and centroid distance are made into hyperparameters to generate a pareto score
- the top 20 pareto score elements are then displayed to the user in the form of an output.csv file
