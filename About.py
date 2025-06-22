import streamlit as st

st.set_page_config(page_title="Welcome | DBS Vision App", page_icon="🩸")

st.title("Welcome to the DBS Vision App")

st.markdown("""
The DBS Vision App allows you to perform computer vision analysis of dried blood spot size and shape, using images obtained using a Revvity Panthera puncher.

Use the sidebar to navigate between pages:
- Understanding how to configure the app and determine the instrument specific pixel-mm conversion factor
- Uploading and analysing single images of dried blood spots
- Exploring additional analysis options (coming soon)



---

### About

This tool was developed by [Nick Flynn](https://www.linkedin.com/in/flynnn) of Cambridge University Hospitals NHS Foundation Trust.

Please cite the following publication if you use this Streamlit app:
             
**A computer vision approach to the assessment of dried blood spot size and quality in newborn screening.**
Flynn N, Moat SJ, Hogg SL.
*Clin Chim Acta.* 2023;547:117418.  
[Read the paper](https://doi.org/10.1016/j.cca.2023.117418)

For questions or comments please contact: [nick.flynn@nhs.net](mailto:nick.flynn@nhs.net)
            
### ⚠️ Disclaimer

This application is intended for research and service evaluation only.

Although this tool was developed using images exported from a **Revvity Panthera puncher**, it is **not developed, endorsed, or supported by Revvity**.

Use of this software is at your own risk. The developer and affiliated institutions accept no responsibility for any consequences arising from its use.
            

""")