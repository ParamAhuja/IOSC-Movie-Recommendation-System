# **IOSC-Movie-Recommendation-System**

The movie recommendation system is a type of information filtering system that suggests movies to the user based on their preference or the rating that they would give to a movie. 


Almost every major tech company has applied them in some form or the other: Amazon uses it to suggest products to customers, YouTube uses it to decide which video to play next on autoplay, and Facebook uses it to recommend pages to like and people to follow. Moreover,  companies like Netflix and Spotify  depend highly on the effectiveness of their recommendation engines for their business and succees.

# Datasets: 
There are two datasets inside the input folder:
1. tmdb-movie-metadata: Metadata on ~5,000 movies from TMDb (43.6MB)
2. the-movies-dataset: Metadata on over 45,000 movies. 26 million ratings from over 270,000 users.(900MB)
   
Source- https://drive.google.com/drive/folders/1QbUZH105H4Rq7Uu5L0cbkTqLp4tbJj17?usp=sharing

More information in the datasets file.


# Working :

There are basically three types of recommender systems:-
> *  **1.Demographic Filtering**- They offer generalized recommendations to every user, based on movie popularity and/or genre. The System recommends the same movies to users with similar demographic features. Since each user is different , this approach is considered to be too simple. The basic idea behind this system is that movies that are more popular and critically acclaimed will have a higher probability of being liked by the average audience.
> * 
I'll be using IMDB's weighted rating (wr) which is given as :-

![](https://image.ibb.co/jYWZp9/wr.png)
where,
* v is the number of votes for the movie;
* m is the minimum votes required to be listed in the chart;
* R is the average rating of the movie; And
* C is the mean vote across the whole report

We already have v(**vote_count**) and R (**vote_average**) and C can be calculated


Under the Trending Now tab of these systems we find movies that are very popular and they can just be obtained by sorting the dataset by the popularity column.


> *  **2.Content Based Filtering**- They suggest similar items based on a particular item. This system uses item metadata, such as genre, director, description, actors, etc. for movies, to make these recommendations. The general idea behind these recommender systems is that if a person liked a particular item, he or she will also like an item that is similar to it.

We will compute pairwise similarity scores for all movies based on their plot descriptions and recommend movies based on that similarity score. The plot description is given in the **overview** feature of our dataset. 
We see that over 20,000 different words were used to describe the 4800 movies in our dataset.

With this matrix in hand, we can now compute a similarity score. There are several candidates for this; such as the euclidean, the Pearson and the [cosine similarity scores](https://en.wikipedia.org/wiki/Cosine_similarity). There is no right answer to which score is the best. Different scores work well in different scenarios and it is often a good idea to experiment with different metrics.

We will be using the cosine similarity to calculate a numeric quantity that denotes the similarity between two movies. We use the cosine similarity score since it is independent of magnitude and is relatively easy and fast to calculate. Mathematically, it is defined as follows:
![](data:image/png;base64,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)


While our system has done a decent job of finding movies with similar plot descriptions, the quality of recommendations is not that great. "The Dark Knight Rises" returns all Batman movies while it is more likely that the people who liked that movie are more inclined to enjoy other Christopher Nolan movies. This is something that cannot be captured by the present system.


> *  **3.Collaborative Filtering**- This system matches persons with similar interests and provides recommendations based on this matching. Collaborative filters do not require item metadata like its content-based counterparts.


# **Collaborative Filtering**

Our content based engine suffers from some severe limitations. It is only capable of suggesting movies which are close to a certain movie. That is, it is not capable of capturing tastes and providing recommendations across genres.

Also, the engine that we built is not really personal in that it doesn't capture the personal tastes and biases of a user. Anyone querying our engine for recommendations based on a movie will receive the same recommendations for that movie, regardless of who she/he is.

Therefore, in this section, we will use a technique called Collaborative Filtering to make recommendations to Movie Watchers.
It is basically of two types:-

*  **User based filtering**-  These systems recommend products to a user that similar users have liked. For measuring the similarity between two users we can either use pearson correlation or cosine similarity.
This filtering technique can be illustrated with an example. In the following matrixes, each row represents a user, while the columns correspond to different movies except the last one which records the similarity between that user and the target user. Each cell represents the rating that the user gives to that movie. Assume user E is the target.
![](https://cdn-images-1.medium.com/max/1000/1*9NBFo4AUQABKfoUOpE3F8Q.png)

Since user A and F do not share any movie ratings in common with user E, their similarities with user E are not defined in Pearson Correlation. Therefore, we only need to consider user B, C, and D. Based on Pearson Correlation, we can compute the following similarity.
![](https://cdn-images-1.medium.com/max/1000/1*jZIMJzKM1hKTFftHfcSxRw.png)

From the above table we can see that user D is very different from user E as the Pearson Correlation between them is negative. He rated Me Before You higher than his rating average, while user E did the opposite. Now, we can start to fill in the blank for the movies that user E has not rated based on other users.
![](https://cdn-images-1.medium.com/max/1000/1*9TC6BrfxYttJwiATFAIFBg.png)

Although computing user-based CF is very simple, it suffers from several problems. One main issue is that users’ preference can change over time. It indicates that precomputing the matrix based on their neighboring users may lead to bad performance. To tackle this problem, we can apply item-based CF.

* **Item Based Collaborative Filtering** - Instead of measuring the similarity between users, the item-based CF recommends items based on their similarity with the items that the target user rated. Likewise, the similarity can be computed with Pearson Correlation or Cosine Similarity. The major difference is that, with item-based collaborative filtering, we fill in the blank vertically, as oppose to the horizontal manner that user-based CF does. The following table shows how to do so for the movie Me Before You.
![](https://cdn-images-1.medium.com/max/1000/1*LqFnWb-cm92HoMYBL840Ew.png)

It successfully avoids the problem posed by dynamic user preference as item-based CF is more static. However, several problems remain for this method. First, the main issue is ***scalability***. The computation grows with both the customer and the product. The worst case complexity is O(mn) with m users and n items. In addition, ***sparsity*** is another concern. Take a look at the above table again. Although there is only one user that rated both Matrix and Titanic rated, the similarity between them is 1. In extreme cases, we can have millions of users and the similarity between two fairly different movies could be very high simply because they have similar rank for the only user who ranked them both.


### **Single Value Decomposition**
One way to handle the scalability and sparsity issue created by CF is to leverage a **latent factor model** to capture the similarity between users and items. Essentially, we want to turn the recommendation problem into an optimization problem. We can view it as how good we are in predicting the rating for items given a user. One common metric is Root Mean Square Error (RMSE). **The lower the RMSE, the better the performance**.

Now talking about latent factor you might be wondering what is it ?It is a broad idea which describes a property or concept that a user or an item have. For instance, for music, latent factor can refer to the genre that the music belongs to. SVD decreases the dimension of the utility matrix by extracting its latent factors. Essentially, we map each user and each item into a latent space with dimension r. Therefore, it helps us better understand the relationship between users and items as they become directly comparable. The below figure illustrates this idea.

![](https://cdn-images-1.medium.com/max/800/1*GUw90kG2ltTd2k_iv3Vo0Q.png)
