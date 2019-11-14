# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 18:53:10 2019

@author: makigondesk
"""

def zarts(X):  # 座圧
    X = [[k] for k in X]  # それぞれの要素に配列を被せる
    #print(X[0])
    #X = [X[i].append(i) for i in range(len(X))] 
    for i in range(len(X)): # 元の数列の順番を記憶する
        X[i].append(i)
    X.sort()  # 昇順にソート
    # X = [X[i].append(i) for i in range(len(X))]  # 置き換える数字を追加
    for i in range(len(X)): # 置き換える数字を追加
        X[i].append(i)
    X.sort(key=lambda x:x[1])  # 元の数列の順番に戻す
    ans = [X[i][2] for i in range(len(X))]  # 置き換えた数字を格納していく
    return ans  # 答え

a = [5, 4, 8, 2, 7]
#zarts(a)
print(zarts(a))