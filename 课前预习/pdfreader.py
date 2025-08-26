from pypdf import PdfReader 
reader = PdfReader("课前预习指导V1.1.pdf") 
page = reader.pages[0] # 获取第一页
print(page.extract_text())