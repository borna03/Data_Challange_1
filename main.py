from config import *
import plot_figures
import clean_data

resp_langs = plot_figures.responded_at_lang(airlines['VirginAtlantic']['id_str'])
resp_langs_sort = dict(sorted(resp_langs.items(), key=lambda item: item[1], reverse=True))
print(resp_langs_sort)
