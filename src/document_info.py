# library
import PyPDF2

# info function
def get_info(path):
    pdf = PyPDF2.PdfFileReader(open(path, "rb"))
    info = pdf.getDocumentInfo()
    print("Document Authon: ", info.author)
    print("Document Creator: ", info.creator)
    print("Document Producer: ", info.producer)
    print("Document Title: ", info.title)

path = "dataset/trainResumes/candidate_002.pdf"
get_info(path)