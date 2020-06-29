# Sentiment_Analysis_Using_Rule_based
# NLP_Read_Before_You_Agree

Please click on this: [Watch and Enjoy](https://www.youtube.com/watch?v=4TxpI0vcaI4&feature=youtu.be)

 ![Image description](Osgoode1.png)
 
 ## Objective:
 
This project outlines the concept of a third-party extension or service (henceforth known as the Read Before You Agree, or ‘RBYA’) connected to conventional browsers (such as Google Chrome) to be used in tandem with a consumer signing up for any online-based product or service with an associated ToS agreement. In addition, the RBYA may also be available as a web or app service where consumers can paste their ToS agreements which will help identify punitive and digital privacy clauses that warrant their attention according to user’s preferences and potentially provide friendlier alternative services.

## Scope: 

RBYA transformed ToS agreement to an ethic vector in 17 categories by using NLP techniques, then compared with the user’s ethics vector using cosine similarity so that RBYA can provide recommendations on whether to use the app or which app to use based on user’s comfort level.

 ## Training and Testing Data:

ToS, DR: Click ==> [Terms of Service](https://tosdr.org/)

The reason for using ToS, DR is they provide not only the punitive labels for the sentences in large companies’ ToS, but also provides a topic classification for each sentence. Both punitive and category labels are available from this website.
  
 ## Methodology:
 

 

 ## Model Result:
 ![Image description](Osgoode1.png)

 
 ## Conclusion:
 
To reach a better performance for the trained model, more data needs to be collected. Giving the limited companies’ terms ToS, DR has provided, it is not enough for building a robust model. To proceed, either waiting for more terms being labeled from the same data source or reaching to a different platform.

Should a platform or App be developed in order to combine the ToS vector and user’s vector, where the user can customize their comfort level and preferences through a well-designed survey?  In the current phase, both the user’s survey and the transformation logic for the user’s vector are still in the prototype.

The ultimate platform should be able to do the followings:

<br> 1. Have an inventory of as many companies’ ToS as possible. 
<br> 2. Have input access for users to copy/paste the ToS they want to check. 
<br> 3. The well-designed survey can reflect the user’s punitive comfort zone/preferences.
<br>4. A logic to transform the survey result to a customized user vector.
<br>5. Our current model for vectorizing the given ToS.
<br>6. Our current mathematic logic to compare ToS’s vector and user’s vector and give related recommendations.<br/>
