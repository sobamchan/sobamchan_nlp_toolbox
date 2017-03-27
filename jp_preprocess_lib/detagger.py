#!/usr/bin/env python
# -*- coding:utf-8 -*-

from html.parser import HTMLParser
import re

# HTML 除去
class TagStripper(HTMLParser):
  def __init__(self):
    HTMLParser.__init__(self)
    self.tags = {}

  def strip(self, html):
    self.text = u''
    self.stack = []
    self.last = None
    try:
        self.feed(html)
        self.close()
    except:
        return html
    return self.text

  def add_sentence_end(self):
    if len(self.text) > 0 \
      and self.text[-1] != u'、' \
      and self.text[-1] != u'。':
        self.text += u"。"

  def handle_starttag(self, tag, args):
    self.stack.append((tag, args))
    # 改行指示が連続していれば文章は切れているはず
    if (self.last in [u'p', u'br'] and tag in [u'p', u'br']):
      self.add_sentence_end()
    # Tag使用状況確認
    self.last = tag
    self.tags[tag] = True

  def handle_endtag(self, tag):
    # 明らかに次の要素に文章が続かないもの
    flag = tag in [u'h1', u'h2', u'h3', u'h4', u'h5', u'h6',
               u'section', u'article', u'blockquote', u'div',
               u'table', u'tbody', u'th', u'td', u'ol', u'li',
               u'iframe',
               u'img', u'audio', u'object', u'figure', u'embed',
               u'center']
    # 確認用
    if len(self.stack) > 0 and tag == self.stack[-1][0]:
      (start_tag, args) = self.stack.pop()
      if len(args) > 0:
        # Noteでの名前タグでは文章は切れているはず
        flag = flag or (tag == u'p' and args[0][0] == u'name')
    # 読点がなくても、文章の最後とみなして読点を追加し文末にする
    if flag:
      self.add_sentence_end()
    self.last = tag

  def handle_data(self, data):
    def is_ascii(s):
      return all(ord(c) < 128 for c in s)
    if is_ascii(data):
      data = data.encode('utf-8')
    if self.last != 'data':
      data = data.strip()
    if len(data) != 0:
      self.text += data
      self.last = 'data'

def detag(text):
  stripper = TagStripper()
  return stripper.strip(text)

# テスト
def test_TagStripper():
  sample = u"""
<h1>タイトル</h1>
文章の途中で<br>
      改行がされていることがある。
でも、完全に行が空いていたら読点がなくても行末だとわかるはず<br>
<p name="a">
それと名前付きpタグの閉じタグも行末だとわかる</p>
"""
  y = u"""タイトル。文章の途中で改行がされていることがある。
でも、完全に行が空いていたら読点がなくても行末だとわかるはず。それと名前付きpタグの閉じタグも行末だとわかる。"""
  x = detag(sample)

if __name__ == "__main__":
    test_TagStripper()
