import streamlit as st
import pickle
import re
#import nltk
import nbimporter
import nbimporter
from Resume_Screening_with_Python import cleanResume
#nltk.download('punkt')
#nltk.download('stopwords')


#loading models
clf = pickle.load(open('clf.pkl','rb'))
tfidfd = pickle.load(open('tfidf.pkl','rb'))

#Web app
def main():
    st.set_page_config(
        page_title="Resume Screening App",
        page_icon=":clipboard:", 
        layout="wide"
    )
    st.title("Resume Screening App")
    st.write("Welcome to the Resume Screening App")
    # Ajouter une icône à côté du titre
    st.markdown(":rocket: **Get your dream job!**")
    st.image("./job.jpg",  use_column_width=True)

    st.write('Upload your resume and get the predicted job category.')

    # Afficher une image à côté du formulaire de téléchargement

    upload_file=st.file_uploader(' Choose a resume file',type=['txt','pdf'])
    
    if upload_file is not None:
        try :
            resume_bytes=upload_file.read()
            resume_text=resume_bytes.decode('utf-8')
        except:
            # If UTF-8 decoding fails, try decoding with 'latin-1'
            resume_text = resume_bytes.decode('latin-1')    
        cleaned_resume = cleanResume(resume_text)
        input_features = tfidfd.transform([cleaned_resume])
        prediction_id = clf.predict(input_features)[0]
       # st.write(prediction_id)
        # Map category ID to category name
        category_mapping = {
            15: "Java Developer",
            23: "Testing",
            8: "DevOps Engineer",
            20: "Python Developer",
            24: "Web Designing",
            12: "HR",
            13: "Hadoop",
            3: "Blockchain",
            10: "ETL Developer",
            18: "Operations Manager",
            6: "Data Science",
            22: "Sales",
            16: "Mechanical Engineer",
            1: "Arts",
            7: "Database",
            11: "Electrical Engineering",
            14: "Health and fitness",
            19: "PMO",
            4: "Business Analyst",
            9: "DotNet Developer",
            2: "Automation Testing",
            17: "Network Security Engineer",
            21: "SAP Developer",
            5: "Civil Engineer",
            0: "Advocate",
        }
        category_name = category_mapping.get(prediction_id, "Unknown")

        st.write(":star: Predicted Job Category:", category_name)
        feedback = st.selectbox("How would you rate this application?", ["Excellent", "Good", "Average", "Poor"])
        st.write(":thumbsup: Thank you for your feedback!")
     
#python main 
if __name__ == "__main__":
    main()    
