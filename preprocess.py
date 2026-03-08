import pandas as pd
import spacy
import string
import tqdm


class Collection:
    """
    Этот класс создает коллекцию объектов типа Document, в которой каждый элемент
    -- это пара (индекс, текст) из файла films_data.csv. По этой коллекции документов
    будет осуществляться поиск.
    """

    def __init__(self):
        N = 1500

        df = (
            pd.read_csv("films_data.csv")
            .sample(N, random_state=42)
            .reset_index(drop=True)
        )

        self.processer = spacy.load("ru_core_news_sm")

        print("Создается и обрабатывается коллекция текстов.")
        self.documents = {
            index: Document(
                self.processer,
                *list(
                    map(
                        str,
                        [
                            row["title"],
                            row["genre"],
                            row["imdb_rating"],
                            row["plot"],
                        ],
                    )
                ),
            )
            for index, row in tqdm.tqdm(df.iterrows())
        }
        print("Коллекция текстов создана.\n")


class Document:
    """
    Этот класс создает объект типа Document, который состоит из текста (описания фильма)
    и его мета данных (название, жанр, рейтинг фильма, длина текста). Каждый текст проходит
    предобработку при помощи метода preprocess (токенизация + лемматизация, удаление стоп-слов
    и пунтуации).
    Для каждого текста хранится список предобработанных токенов.
    В классе перегружен метод __str__ для «красивого» вывода результатов поиска.
    """

    def __init__(self, processer, title, genre, rating, text):
        self.title = title
        self.genre = genre
        self.rating = rating
        self.text = text
        self.length, self.tokens = self.preprocess(processer, self.text)
        self.preprocessedText = " ".join(self.tokens)

    def preprocess(self, processer, text):
        """
        Эта функция предобрабатывает текст.
        :argument processer: spacy-обработчик для русского языка
        :argument text: текст, который нужно предобработать
        :returns: длина текста и список лемм
        """
        lemmas = []
        processedText = processer(text)
        for token in processedText:
            if token.lemma_ not in string.punctuation and not token.is_stop:
                lemmas.append(token.lemma_)
        return len(processedText), lemmas

    def __str__(self):
        return f"Название: {self.title}\nЖанр: {self.genre}\nРейтинг на IMDB: {self.rating}\n{self.text}\n"
