import pdfquery
import pandas as pd

fpath = "G:\\Dropbox\\Projects\\Financial_Independence_Project\\statements_pdf\\2019_10.pdf"
output_xm_fpath = "G:\\Dropbox\\Projects\\Financial_Independence_Project\\statements_pdf\\2019_10.xml"

search_columns_df = pd.DataFrame(columns=['label', 'alignment'])


pdf = pdfquery.PDFQuery(fpath)
print('Loading PDF... ', end='')
pdf.load()
print('completed!')

pages = pdf.tree.xpath('//*/LTPage')
list_objs = pdf.tree.xpath('//*/LTPage/LTImage/LTFigure/LTFigure/LTTextBoxHorizontal/LTTextLineHorizontal')
paths2 = pdf.tree.xpath('//*/LTPage/LTImage/LTFigure/LTFigure/LTTextLineHorizontal')
date_objs = [obj for obj in list_objs if obj.layout.get_text().find('Date') != -1]

for date in date_objs:
    print(f"{date.attrib['x0']}")

pdf.tree.write(output_xm_fpath, pretty_print=True, encoding="utf-8")

g = 0