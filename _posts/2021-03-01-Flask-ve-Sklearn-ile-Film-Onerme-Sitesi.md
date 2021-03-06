---
layout: post
title: Flask ve Sklearn ile Film Önerme Sitesi Yapalım 
preview: Metin benzerliği benzerliğini kullanarak Flask ile film önerme sitesi yapalım.
---

**İçerik**
* TOC
{:toc}

Bu yazıdaki bütün kodlar [Bu repodan](https://github.com/ocakhasan/movie-recommender) bulunmaktadır. Eğer demo versiyonunu görmek isterseniz [http://banafilmoner.herokuapp.com/](http://banafilmoner.herokuapp.com/) sitesinden görebilirsiniz.

## Gereksinimler
Bu yazımızda yapacağımız siteyi eğer kendiniz de yapmak istiyorsanız [Flask](https://flask.palletsprojects.com/en/1.1.x/) ve [Scikit-learn](https://scikit-learn.org/) kütüphanelerini yüklemeniz gerekmektedir. Bunları yüklemek için terminalden şu komutları yazabilirsiniz ya da her bir paketin dökümentasyonundan bakabilirsiniz.

```
pip install Flask 
pip install scikit-learn
```

## Sitenin Yapısı
Yapacağımız sitede film önerileri metin benzerliği ile olacak. Bu filmlerin açıklama metinlerini ise bir veri kümesinden alacağız. Bu veri kümesine [TMDB 5000 Movies](https://www.kaggle.com/tmdb/tmdb-movie-metadata) sayfasından ulaşabilirsiniz. Bundan dolayı önerebileceğimiz metinler sadece bu veri kümesindekiler olacaktır. Metin benzerliğini ise [kosinüs benzerliği](https://merveenoyan.medium.com/kosin%C3%BCs-benzerli%C4%9Fi-2b4a4c924f27) ile yapacağız. 

## Veri Seti ve Metin Benzerliği
Veri setindeki `title` filmin başlığını ve `overview` ise filmi basitçe açıklayan metin. Biz `overview` sütununu kullanarak metin benzerliğini kuracağız. Bunun için önce `utils.py` diye bir dosya oluşturalım ve indirdiğimiz veri setini de projedeki dosyaya koyalım. Öncelikle kosinüs benzerliğini verecek olan bir fonksiyon yazalım.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

def get_cosine_similarities(df):

    vectorizer = TfidfVectorizer(stop_words="english")

    tf_idf_mat = vectorizer.fit_transform(df['overview'])

    cosine_sim = linear_kernel(tf_idf_mat, tf_idf_mat)

    return cosine_sim
```

`get_cosine_similarities(df)` fonksiyonu parametere olarak `DataFrame` alır, `DataFrame`i ise dosyayı okuduktan sonra alıp daha sonra bu fonksiyona parametre olarak vereceğiz. Fonksiyonda kullanılan [`TfidfVectorizer`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) metinlerden bilgi çıkarmamıza yarayan bir algoritmadır. Açılımı `Term frequency (tf) -> (terim sıklığı)` ve `inverse document frequency (ters döküman sıklığı)`. Yani terimlerin her bir metinde ne kadar sıklıkla geçtiğine ve bir de bütün dökümanda ne kadar sıklıkla geçtiğine bakıp, hangi terimlerin cümleleri ayırmada önemli olduğuna karar verir. Bu bize (4803, n) boyutunda bir matrix dönderecektir. `n` ise bu algoritmanın bulduğu belirleyici kelimelerdir. Yani her bir cümle için her bir kelimenin ne kadar önemi var, bunu gösteren bir matrix. Daha sonra bu matrixi kullanarak her bir metin arasındaki benzerliği bulmak için [`linear_kernel`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.linear_kernel.html) kullanıyoruz. Bu algoritma ise bize (4803, 4803) boyutunda bir matrix dönderecek bu da her bir metinin diğer 4038 filmle benzerliğini gösteren bir matrix olacak. 
Bu fonksiyondan çıkan sonuç ise şu şekildedir

```
[[1.         0.         0.         ... 0.         0.         0.        ]
 [0.         1.         0.         ... 0.02160533 0.         0.        ]
 [0.         0.         1.         ... 0.01488159 0.         0.        ]
 ...
 [0.         0.02160533 0.01488159 ... 1.         0.01609091 0.00701914]
 [0.         0.         0.         ... 0.01609091 1.         0.01171696]
 [0.         0.         0.         ... 0.00701914 0.01171696 1.        ]]
```
Görüldüğü gibi bazı değerler 0 bazıları 1 (köşegendekiler), bazıları da 0 ile 1 arasında. Bu demek oluyor ki 0 olanlar arasında hiçbir benzerlik yok, 1 olanlar zaten kendileri ile ölçüldüğü için, örnek olarak 1.film ile 1.film arasındaki benzerlik 1 olacak doğal olarak, 0-1 arasındakiler ise iki film arasındaki benzerliği gösteriyor. 

Ne yapabildiğimizi kısaca yazalım. 
* Veri setini okuduk
* Kosinüs benzerlik matriksini oluşturduk.

Şimdi yapmamız gerekenler ise bize bu matrixi kullanıp belirli bir film için önerilen filmleri döndürebilmek. Bunun için yapmamız gerekenler
1. Kosinüs matrixini kullanıp bize verilen film için önerileri döndüren bir fonksiyon yazmak
2. Flask ile web arayüzü oluşturup, kullanıcın girdiği filme öneriler vermek
3. Bu fonksiyonu flask ile bağlayabilmek.

### Film Önerme Fonksiyonu
Bu fonksiyona geçmeden önce veriyi okuyalım, ve kosinüs matriximizi alalım. Şunu belirtmem gerekir ki, kullanıcının attığı her requestte bu veriyi okuyup kosinüs matrixini okumak saçma olur. Bundan dolayı bunu bir kez yapmak adına yazacağımız kodu `if __name__ == "__main__` altına yazacağız.

Öncelikle bir `app.py` adında bir dosya açalım. Bu dosyada Flask applikasyonumuzun kodları olacak. Diğer `utils.py` dan fonksiyonları çağıracağız. 

`app.py` dosyasına şu kodları girelim. 

```python
from flask import Flask, render_template, request, redirect, flash, url_for
import pandas as pd
import utils

app = Flask(__name__)

if __name__== "__main__":
    df = pd.read_csv("data.csv")
    df['overview'] = df['overview'].fillna('')
    df['lower_name'] = df['title'].str.lower()

    titles = pd.Series(df.index, index=df['lower_name']).drop_duplicates()

    cosine_sim = utils.get_cosine_similarities(df)

    app.run()
```

Şuan `app.py` dosyasında yaptığımız işlemler.
1. Flask uygulaması oluşturduk.
2. Veriyi okuduk.
3. Kosinüs benzerlik matriksini aldık.

Main kısmında ***titles*** diye bir değişken oluşturma sebebimiz bunu filmleri önerecek olan fonksiyonda kullanacağımızdan dolayıdır. ***titles*** değişkeni tip olarak `Series`dir. Konsola yazdırdığımız zaman şöyle bir sonuç çıkacaktır. 

```
lower_name
avatar                                         0
pirates of the caribbean: at world's end       1
spectre                                        2
the dark knight rises                          3
john carter                                    4
                                            ...
el mariachi                                 4798
newlyweds                                   4799
signed, sealed, delivered                   4800
shanghai calling                            4801
my date with drew                           4802
Length: 4803, dtype: int64
```

Şimdi filmleri önerecek fonksiyonu yazmaya başlayabiliriz. Bunu `utils.py` dosyasında yazalım. 

```python
"""
movie_title = istenilen filmin ismi
cosine_similarity = kosinüs benzerlik matriksi
titles= az önce oluşturduğumuz filmin isimlerine sahip olan `Series`
df = bütün filmleri barındıran dataframe
"""
def get_recommendations(movie_title, cosine_similarity, titles, df):

    index_movie = titles[movie_title]                   #istenilen filmin indexini bul
    name_of_movie = df.iloc[index_movie]['title']       #daha sonra dataframeden filmin adını bul. 
                                                        #istenilen isim küçük harfli olabilir, biz
                                                        #dataframde nasılsa onu almak için yapıyoruz.
    
    similarities = cosine_similarity[index_movie]       #daha sonra girilen filmin kosinüs benzerlik 
                                                        #arrayini al, diğer filmlerle benzerlik arrayi

    similarity_scores = list(enumerate(similarities))   #işlem kolaylığı için her bir benzerliğin indexini
                                                        #alabilmemiz lazım. yani (0, 0.2), (1, 0.4), (2. 0.7) ... gibi.
    similarity_scores = sorted(similarity_scores , key=lambda x: x[1], reverse = True) #bütün benzerlik skorlarını sırala
    similarity_scores = similarity_scores[1:11]         #en benzer 10 filmi al
    similar_indexes = [x[0] for x in similarity_scores] #benzer filmlerin indexlerini al
    
    return df.iloc[similar_indexes], name_of_movie      #benzer filmlerin bilgilerini almak için indexlerini kullan.
```

Örnek olarak `get_recommendations("avatar", cosine_sim, titles, df)` çağırırsak çıkacak sonuç
<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>overview</th>
      <th>id</th>
      <th>homepage</th>
      <th>release_date</th>
      <th>runtime</th>
      <th>lower_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3604</th>
      <td>Apollo 18</td>
      <td>Officially, Apollo 17 was the last manned miss...</td>
      <td>50357</td>
      <td>http://apollo18movie.net/</td>
      <td>2011-07-20</td>
      <td>86.0</td>
      <td>apollo 18</td>
    </tr>
    <tr>
      <th>2130</th>
      <td>The American</td>
      <td>Dispatched to a small Italian town to await fu...</td>
      <td>27579</td>
      <td>http://focusfeatures.com/film/the_american/</td>
      <td>2010-08-31</td>
      <td>104.0</td>
      <td>the american</td>
    </tr>
    <tr>
      <th>634</th>
      <td>The Matrix</td>
      <td>Set in the 22nd century, The Matrix tells the ...</td>
      <td>603</td>
      <td>http://www.warnerbros.com/matrix</td>
      <td>1999-03-30</td>
      <td>136.0</td>
      <td>the matrix</td>
    </tr>
    <tr>
      <th>1341</th>
      <td>The Inhabited Island</td>
      <td>On the threshold of 22nd century, furrowing th...</td>
      <td>16911</td>
      <td>http://oostrov.ru</td>
      <td>2008-12-18</td>
      <td>115.0</td>
      <td>the inhabited island</td>
    </tr>
    <tr>
      <th>529</th>
      <td>Tears of the Sun</td>
      <td>Navy SEAL Lieutenant A.K. Waters and his elite...</td>
      <td>9567</td>
      <td>NaN</td>
      <td>2003-03-07</td>
      <td>121.0</td>
      <td>tears of the sun</td>
    </tr>
    <tr>
      <th>1610</th>
      <td>Hanna</td>
      <td>A 16-year-old girl raised by her father to be ...</td>
      <td>50456</td>
      <td>http://hannathemovie.com/</td>
      <td>2011-04-07</td>
      <td>111.0</td>
      <td>hanna</td>
    </tr>
    <tr>
      <th>311</th>
      <td>The Adventures of Pluto Nash</td>
      <td>The year is 2087, the setting is the moon. Plu...</td>
      <td>11692</td>
      <td>NaN</td>
      <td>2002-08-15</td>
      <td>95.0</td>
      <td>the adventures of pluto nash</td>
    </tr>
    <tr>
      <th>847</th>
      <td>Semi-Pro</td>
      <td>Jackie Moon is the owner, promoter, coach, and...</td>
      <td>13260</td>
      <td>http://newline.com/properties/semipro.html</td>
      <td>2008-02-28</td>
      <td>91.0</td>
      <td>semi-pro</td>
    </tr>
    <tr>
      <th>775</th>
      <td>Supernova</td>
      <td>Set in the 22nd century, when a battered salva...</td>
      <td>10384</td>
      <td>NaN</td>
      <td>2000-01-14</td>
      <td>91.0</td>
      <td>supernova</td>
    </tr>
    <tr>
      <th>2628</th>
      <td>Blood and Chocolate</td>
      <td>A young teenage werewolf is torn between honor...</td>
      <td>10075</td>
      <td>NaN</td>
      <td>2007-01-26</td>
      <td>98.0</td>
      <td>blood and chocolate</td>
    </tr>
  </tbody>
</table>
</div>


### HTML Arayüz
Bu fonksiyonu da yazdığımıza göre şimdi Flask ile bağlayabiliriz. Ama öncelikle bir arayüzümüz olması gerekiyor. Bunun için aynı klasörde `templates` diye bir klasör oluşturun ve içine `index.html` adında bir dosya oluşturun. Bu dosya bizim kullanıcıdan arayüzü almamızı sağlayacak olan `HTML` kodunu içerecek. `HTML` kısmını anlatmayacağım. Basit şekilde `Flask` bildiğinizi varsayıyorum. 

```html
<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Recommend Me a Movie</title>
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css"
        integrity="sha384-JcKb8q3iqJ61gNV9KGb8thSsNjpSL0n8PARn9HuZOnIxN0hoP+VmmDGMN5t9UJ0Z" crossorigin="anonymous">
    <link href="https://fonts.googleapis.com/css2?family=Roboto&display=swap" rel="stylesheet">
</head>

<body>
    <style>
        body {
            font-family: 'Roboto', sans-serif;
        }
    </style>

    <div class="container mt-3">
        <nav class="navbar navbar-expand-lg navbar-light bg-light">
            <a class="navbar-brand" href="#">
                <img src="{{url_for('static', filename='cinema.svg')}}" width="30" height="30"
                    class="d-inline-block align-top" alt="" loading="lazy">
                Movie Recommender
            </a>
        </nav>


        <form action="/" method="POST">
            <div class="input-group mb-3 mt-3">
                <div class="input-group-prepend">
                    <span class="input-group-text" id="basic-addon1">@</span>
                </div>
                <input type="text" class="form-control" placeholder="Movie" aria-label="movie_name"
                    aria-describedby="basic-addon1" name="fname" id="fname">
                <input type="submit" value="Submit" class="btn btn-outline-primary">
            </div>
        </form>


        {% if length %}
        <p>Recommendations for <strong>{{movie_name}}</strong> </p>
        <ul class="list-group list-group-flush">
            {% for i in range(length) %}
            <div class="card mb-3">
                <div class="card-header">
                    <h4>{{context.movies[i]}}</h4>
                </div>
                <div class="card-body">
                    <blockquote class="blockquote mb-0">
                        <p>{{context.overviews[i]}}</p>
                    </blockquote>
                    <p class="card-text"><strong>Runtime :</strong> {{context.runtimes[i]}} minutes</p>
                    <p class="card-text"><strong>Release Date :</strong> {{context.release_dates[i]}}</p>
                </div>
            </div>
            {% endfor %}
        </ul>

        {% endif%}

        {% if error %}
        <h4>Could not find the movie. Please check the input, if necessary write the whole name of movie and try again.
        </h4>
        {% endif %}

    </div>


    <script src="https://code.jquery.com/jquery-3.5.1.slim.min.js"
        integrity="sha384-DfXdz2htPH0lsSSs5nCTpuj/zy4C+OGpamoFVy38MVBnE+IbbVYUew+OrCXaRkfj"
        crossorigin="anonymous"></script>
    <script src="https://cdn.jsdelivr.net/npm/popper.js@1.16.1/dist/umd/popper.min.js"
        integrity="sha384-9/reFTGAW83EW2RDu2S0VKaIzap3H66lZH81PoYlFhbGU+6BZp6G7niu735Sk7lN"
        crossorigin="anonymous"></script>
    <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/js/bootstrap.min.js"
        integrity="sha384-B4gt1jrGC7Jh4AgTPSdUtOBvfO8shuf57BaghqFfPlYxofvL8/KUEfYiJOMMV+rV"
        crossorigin="anonymous"></script>




</body>

</html>
```
### Flask Endpointleri halletme
Bu kodda dikkatinizi çekmek istediğim bir nokta var. `FORM` bir '/' yoluna **POST** request yapıyor. Bizim uygulamamızda bir endpoint olacak ve bu da giriş sayfası. Hem `GET` hem de `POST` requestler buraya atılacak. Şimdi `app.py` dosyasında bu koşulları sağlayan kodumuzu yazalım. 

```python
from flask import Flask, render_template, request, redirect, flash, url_for
import pandas as pd
import utils

app = Flask(__name__)



@app.route('/', methods=['GET', 'POST'])
def hello():
    length = 0
    movie_name = ""
    context = {                     #Bu dictionary önerilen filmlerin bilgilerini tutuyor.
        'movies': [],               #isimler
        'urls': [],                 #filmlerin sayfaları
        'release_dates': [],        #filmlerin yayınlanma tarihleri
        'runtimes': [],             #filmlerin süreleri
        'overviews': []             #filmleri anlatan metinler
    }

    if request.method == "POST":                    #Kullanıcı bir input girdiyse
        text = request.form['fname'].lower()
        print("request text", text)
        try:
            recommended_df, movie_name = utils.get_recommendations(
                text, cosine_sim, titles, df)                               #girilen inputtan filmleri al
            context['movies'] = recommended_df.title.values
            context['urls'] = recommended_df.homepage.values
            context['release_dates'] = recommended_df.release_date.values
            context['runtimes'] = recommended_df.runtime.values
            context['overviews'] = recommended_df.overview.values

            length = len(context['movies'])
        except:
            return render_template('index.html', error=True)            #filmi bulamadıysak error döndür.

    return render_template('index.html', length=length, context=context, movie_name=movie_name, error=False) 


if __name__ == '__main__':
    df = pd.read_csv("data.csv")
    df['overview'] = df['overview'].fillna('')

    titles = pd.Series(df.index, index=df['lower_name']).drop_duplicates()

    cosine_sim = utils.get_cosine_similarities(df)

    app.run()

```
Render templatede gönderdiğimiz `context` değişkeni `HTML` dosyasında parse ediliyor ve bilgiler güzel bir şekilde gösteriliyor. Dediğim gibi basit şekilde *Flask* bildiğiniz düşünüyorum.


Bundan sonra yapmanız gereken işlem sadece bu dosyayı çalıştırıp kendiniz test edebilirsiniz. Beğendiyseniz paylaşırsanız çok sevinirim. İyi öğrenmeler.
