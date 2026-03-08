import numpy as np
import math
from sklearn.feature_extraction.text import CountVectorizer
from bm25_vectorizer import BM25Vectorizer
import tqdm
from preprocess import *


class Search:
    """
    Это базовый класс, от которого будут наследоваться классы с поиском.
    Он создает коллекцию текстов, множество уникальных слов, считает среднюю длину
    документов в корпусе и их количество.
    """

    def __init__(self):
        self.collection = Collection()
        self.processer = self.collection.processer
        self.documents = self.collection.documents
        self.N = len(self.documents)

        self.vocabulary = set()
        for doc in self.documents:
            self.vocabulary.update(self.documents[doc].tokens)

        self.meanLength = (
            sum([self.documents[doc].length for doc in self.documents]) / self.N
        )


class SearchDefault(Search):
    """
    Класс, в котором осуществляется поиск при помощи готовых реализаций обратного
    индекса через частоты (CountVectorizer из библиотеки sklearn) и через BM-25
    (BM25Vectorizer из библиотеки bm25_vectorizer).
    """

    def __init__(self):
        super().__init__()

        print("Осуществляется векторизация текстов.")
        self.cnt_vectorizer = CountVectorizer()
        self.cnt_matrix = self.cnt_vectorizer.fit_transform(
            [doc.preprocessedText for doc in tqdm.tqdm(self.documents.values())]
        )

        self.bm25_vectorizer = BM25Vectorizer()
        self.bm25_matrix = self.bm25_vectorizer.fit_transform(
            [doc.preprocessedText for doc in tqdm.tqdm(self.documents.values())]
        )
        print("Векторизация текстов завершена.\n")

    def queryProcess(self, query, indexType):
        """
        Эта функция предобрабатывает запрос и возвращает нужную для поиска
        матрицу в зависимости от типа индекса, который ввел пользователь.
        :argument query: запрос от пользователя
        :argument indexType: тип индекса («по частотам» или «BM-25»)
        :returns: предобработанный и векторизованный запрос и матрица для поиска
        """
        query = [q.lemma_ for q in self.processer(query)]

        if indexType == "по частотам":
            return self.cnt_vectorizer.transform(query), self.cnt_matrix
        if indexType == "BM-25":
            return self.bm25_vectorizer.transform(query), self.bm25_matrix

    def search(self, query, indexType):
        """
        Функция, которая осуществляет поиск по запросу среди коллекции текстов.
        :argument query: запрос от пользователя
        :argument indexType: тип индекса («по частотам» или «BM-25»)
        :returns: список отранжированных результатов поиска
        """
        query_v, matrix = self.queryProcess(query, indexType)
        results = (matrix.todense() @ query_v.todense().T).tolist()

        rank = sorted(
            [(idx, value[0]) for idx, value in zip(self.documents, results)],
            key=lambda x: x[1],
            reverse=True,
        )

        return [self.documents[idx] for (idx, value) in rank if value > 0]


class SearchDict(Search):
    """
    Класс, в котором осуществляется поиск при помощи обратного
    индекса через частоты и BM-25, реализованных вручную на основе словарей.
    """

    def __init__(self):
        super().__init__()

        print("Осуществляется создание словарей.")
        self.freqDict = self.invertedIndexFrequency()
        self.bm25Dict = self.invertedIndexBM25()
        print("Создание словарей завершено.\n")

    def invertedIndexFrequency(self):
        """
        Функция, которая создает словарь, где для каждого уникального слова
        хранится список из пар (индекс документа, частота слова в этом документе)
        для всех документов, в которых слово встретилось.
        """
        d = {}
        for token in tqdm.tqdm(self.vocabulary):
            for idx in self.documents:
                if token not in d:
                    d[token] = []
                freq = self.documents[idx].tokens.count(token)
                if freq > 0:
                    d[token].append((idx, freq))
        return d

    def invertedIndexBM25(self):
        """
        Функция, которая создает словарь, где для каждого уникального слова
        хранится список из пар (индекс документа, BM-25 для этого слова и документа)
        для всех документов, в которых слово встретилось.
        Формула взята из семинарской презентации.
        """
        d = {}

        for token in tqdm.tqdm(self.vocabulary):
            for idx, freq in self.freqDict[token]:
                if token not in d:
                    d[token] = []
                tf = freq
                df = len(self.freqDict[token])
                idf = math.log(self.N / df, math.e)
                k = 2
                b = 0.75
                value = (
                    idf
                    * (tf * (k + 1))
                    / (
                        tf
                        + k * (1 - b + b * self.documents[idx].length / self.meanLength)
                    )
                )
                d[token].append((idx, value))
        return d

    def queryProcess(self, query, indexType):
        """
        Эта функция предобрабатывает запрос и возвращает нужный для поиска
        словарь в зависимости от типа индекса, который ввел пользователь.
        :argument query: запрос от пользователя
        :argument indexType: тип индекса («по частотам» или «BM-25»)
        :returns: предобработанный запрос и словарь для поиска
        """
        query = [q.lemma_ for q in self.processer(query)]

        if indexType == "по частотам":
            return query, self.freqDict
        if indexType == "BM-25":
            return query, self.bm25Dict

    def search(self, query, indexType):
        """
        Функция, которая осуществляет поиск по запросу среди коллекции текстов.
        :argument query: запрос от пользователя
        :argument indexType: тип индекса («по частотам» или «BM-25»)
        :returns: список отранжированных результатов поиска
        """
        query, dict_ = self.queryProcess(query, indexType)

        rank = {}
        for q in query:
            if q in dict_:
                for doc in dict_[q]:
                    if doc[0] not in rank:
                        rank[doc[0]] = 0
                    rank[doc[0]] += doc[1]

        return [
            self.documents[idx]
            for idx in sorted(rank, key=lambda x: rank[x], reverse=True)
            if rank[idx] > 0
        ]


class SearchMatrix(Search):
    """
    Класс, в котором осуществляется поиск при помощи обратного
    индекса через частоты и BM-25, реализованных вручную на основе матричных вычислений.
    """

    def __init__(self):
        super().__init__()

        self.idx2word = {
            i: word for (i, word) in zip(range(len(self.vocabulary)), self.vocabulary)
        }

        print("Осуществляется создание матриц.")
        self.freqMatrix = self.invertedIndexFrequency()
        self.bm25Matrix = self.invertedIndexBM25()
        print("Создание матриц завершено.\n")

    def invertedIndexFrequency(self):
        """
        Функция, которая создает матрицу, где для каждой пары (слово, документ)
        хранится, сколько раз это слово встретилось в этом документе.
        """
        matrix = np.zeros((self.N, len(self.vocabulary)))

        for idx_w in tqdm.tqdm(self.idx2word):
            for idx_d in self.documents:
                matrix[idx_d][idx_w] = self.documents[idx_d].tokens.count(
                    self.idx2word[idx_w]
                )

        return matrix

    def invertedIndexBM25(self):
        """
        Функция, которая создает матрицу, где для каждой пары (слово, документ)
        хранится BM-25 этого слова для этого документа.
        """
        matrix = np.zeros((self.N, len(self.vocabulary)))

        for idx_w in tqdm.tqdm(self.idx2word):
            for idx_d in self.documents:
                tf = self.freqMatrix[idx_d][idx_w]
                df = np.count_nonzero(self.freqMatrix[:, idx_w])
                idf = math.log(self.N / df, math.e)
                k = 2
                b = 0.75
                matrix[idx_d][idx_w] = idf * (
                    tf
                    * (k + 1)
                    / (
                        tf
                        + k
                        * (1 - b + b * self.documents[idx_d].length / self.meanLength)
                    )
                )

        return matrix

    def queryProcess(self, query, indexType):
        """
        Эта функция предобрабатывает запрос и возвращает нужную для поиска
        матрицу в зависимости от типа индекса, который ввел пользователь.
        :argument query: запрос от пользователя
        :argument indexType: тип индекса («по частотам» или «BM-25»)
        :returns: предобработанный и векторизованный запрос и матрица для поиска
        """
        queryLemmatized = [q.lemma_ for q in self.processer(query)]
        queryVector = np.zeros(len(self.vocabulary))

        for i in self.idx2word:
            if self.idx2word[i] in queryLemmatized:
                queryVector[i] = queryLemmatized.count(self.idx2word[i])

        if indexType == "по частотам":
            return queryVector, self.freqMatrix
        if indexType == "BM-25":
            return queryVector, self.bm25Matrix

    def search(self, query, indexType):
        """
        Функция, которая осуществляет поиск по запросу среди коллекции текстов.
        :argument query: запрос от пользователя
        :argument indexType: тип индекса («по частотам» или «BM-25»)
        :returns: список отранжированных результатов поиска
        """
        query, matrix = self.queryProcess(query, indexType)
        values = matrix @ query.T
        rank = {i: values[i] for i in range(len(self.documents))}

        return [
            self.documents[idx]
            for idx in sorted(rank, key=lambda x: rank[x], reverse=True)
            if rank[idx] > 0
        ]
