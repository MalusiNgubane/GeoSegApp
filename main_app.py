import streamlit as st
from PIL import Image
import base64
from segmentation_model import segmentation_model_page
from height_segmentation_model import height_segmentation_model_page
from company import company_page 

# Function to add background image
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover;
        background-position: center;
    }}
    </style>
    """,
    unsafe_allow_html=True
    )

# Function to create the landing page
def landing_page():
    st.markdown("""
    <div style='text-align: center; padding: 20px; background-color: rgba(0,0,0,0.5);'>
        <h1 style='font-size: 80px; color: white;'>GeoSeg</h1>
        <h2 style='font-size: 40px; color: white;'>Do you want to map the future of Connectivity?</h2>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        if st.button("Yes", key="yes_button"):
            st.session_state.page = "main"
            st.rerun()
        if st.button("No", key="no_button"):
            st.session_state.page = "goodbye"
            st.rerun()

def goodbye_page():
    st.markdown("""
    <div style='text-align: center; padding: 20px; background-color: rgba(0,0,0,0.5);'>
        <h1 style='color: white;'>Thanks for visiting!</h1>
        <p style='color: white;'>We hope to see you again soon.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        if st.button("Return to Start"):
            st.session_state.page = "landing"
            st.rerun()

def contact_us_page():
    st.markdown("""
    <div style='background-color: rgba(0,0,0,0.5); padding: 20px; border-radius: 10px;'>
        <h1 style='color: white; text-align: center;'>Contact Us</h1>
        <p style='color: white; text-align: center;'>Get in touch with our team</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.form("contact_form"):
        name = st.text_input("Name")
        email = st.text_input("Email")
        message = st.text_area("Message")
        submit_button = st.form_submit_button("Send Message")
        
        if submit_button:
            st.success("Thank you for your message! We'll get back to you soon.")
    
    if st.button("← Back to Main Page"):
        st.session_state.page = "main"
        st.rerun()

# Function to create the main page
def main_page():
    # [Previous styles remain the same]

    # Navigation menu with columns
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown("""
        <div class="dropdown">
            <button class="menu-item">Home</button>
            <div class="dropdown-content">
                <a href="#about-us">About Us</a>
                <a href="#our-mission">Our Mission</a>
                <a href="#clients">Clients</a>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        if st.button("Semantic Segmentation Model", key="seg_model"):
            st.session_state.page = "segmentation_model"
            st.rerun()
    
    with col3:
        if st.button("Height Segmentation Model", key="height_model"):
            st.session_state.page = "height_segmentation_model"
            st.rerun()
    
    with col4:
        if st.button("Company", key="company"):
            st.session_state.page = "company"
            st.rerun()
    
    with col5:
        if st.button("Contact Us", key="contact"):
            st.session_state.page = "contact_us"
            st.rerun()

    # Main content sections
    st.markdown("""
    <div id="about-us" class="section">
        <h2>About Us</h2>
        <p>At GeoSeg, we specialize in cutting-edge geospatial technology to optimize wireless communication networks. 
        Our advanced segmentation models harness the power of satellite and LiDAR data to create precise clutter maps, 
        identifying obstacles such as buildings and vegetation that impact radio frequency propagation. By providing 
        detailed environmental insights, we help telecom operators make informed decisions about network design, 
        tower placement, and signal coverage, all while significantly reducing costs associated with traditional 
        clutter data acquisition.</p>
    </div>
    
    <div id="our-mission" class="section">
        <h2>Our Mission</h2>
        <p>GeoSeg's mission is to revolutionize the way telecom operators approach network planning by providing 
        accurate, cost-effective clutter data through advanced segmentation models. We aim to empower our clients 
        with real-time, actionable insights into their environments, allowing them to optimize signal coverage, 
        improve service quality, and reduce operational costs. By leveraging the latest geospatial technology, 
        we are committed to creating smarter, more reliable communication networks for the future.</p>
    </div>
    
    <div id="clients" class="section">
        <h2>Our Clients</h2>
        <p>Whether you're looking to optimize tower placement, improve coverage, or reduce clutter data costs, 
        we provide the tools and insights needed to make informed, data-driven decisions. Explore our services 
        and see how GeoSeg can support your network optimization needs with precision and reliability.</p>
        <div class="client-logos">
            <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/c/c0/Logotipo_de_Google_Earth.png/1200px-Logotipo_de_Google_Earth.png" alt="Client 1">
            <img src="https://www.tmforum.org/catalysts/_next/image?url=https%3A%2F%2Fmyaccount.tmforum.org%2Fimages%2Fnet_389078%2FCompanyLogo.png&w=3840&q=75" alt="Client 2">
            <img src="https://images.crowdspring.com/blog/wp-content/uploads/2023/07/03162944/amazon-logo-1.png" alt="Client 3">
            <img src="https://freelogopng.com/images/all_img/1686390747tesla-logo-transparent.png" alt="Client 4">
        </div>
    </div>
    """, unsafe_allow_html=True)

# Main app logic
def main():
    st.set_page_config(page_title="GeoSeg", page_icon="🌍", layout="wide")
    
    # Add background image
    add_bg_from_local('point-cloud-data.jpg')
    
    # Initialize session state
    if 'page' not in st.session_state:
        st.session_state.page = "landing"
    
    # Display appropriate page based on session state
    if st.session_state.page == "landing":
        landing_page()
    elif st.session_state.page == "main":
        main_page()
    elif st.session_state.page == "segmentation_model":
        segmentation_model_page()
    elif st.session_state.page == "height_segmentation_model":
        height_segmentation_model_page()
    elif st.session_state.page == "company":
        company_page()
    elif st.session_state.page == "contact_us":
        contact_us_page()
    elif st.session_state.page == "goodbye":
        goodbye_page()

if __name__ == "__main__":
    main()
