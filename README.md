The current challenge is to fix Amazon AWS EC2 server issues and get the LSTM to recognize negation and nuances. The latest file is 'Keras LSTM.ipynb'. Accuracy for that classifier is above 80% but there is still much room for improvement. You can import the trained model by Keras load_model('four_emotions.h5'). You'll also need the wiki vec file to make predictions, but I plan to try a LMDB for much quicker loading of the word vector. 

To see this code in action on the web, go to http://emotionsapi-env.nxhmmmextx.us-east-1.elasticbeanstalk.com/
Note: It hasn't been working as of late. 
To try a proof-of-concept based on a similar API, go to http://emojianalyzer.s3-website-us-east-1.amazonaws.com/. The API used for that detects all four of my detected emotions, plus surprise. As soon as the AWS issues on my end are cleared up we will integrate with that emoji website.

To get the source data used for that LSTM project:

Wiki Vec: https://fasttext.cc/docs/en/english-vectors.html
ISEAR CSV: https://github.com/PoorvaRane/Emotion-Detector/blob/master/ISEAR.csv
Semeval: https://competitions.codalab.org/competitions/17751#learn_the_details-datasets

