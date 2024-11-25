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
        if st.button("No", key="no_button"):
            st.session_state.page = "goodbye"

# Function to create the main page
def main_page():
    # Add JavaScript to handle menu clicks
    st.markdown("""
    <script>
    function navigateTo(page) {
        const event = new CustomEvent('streamlit:setComponentValue', {
            detail: {
                name: 'page',
                value: page
            }
        });
        window.dispatchEvent(event);
    }
    </script>
    """, unsafe_allow_html=True)

    st.markdown("""
    <style>
    .menu {
        display: flex;
        justify-content: center;
        align-items: center;
        background-color: rgba(0,0,0,0.5);
        padding: 10px 0;
    }
    .menu-item {
        color: white;
        text-decoration: none;
        font-size: 18px;
        margin: 0 15px;
        position: relative;
        cursor: pointer;
    }
    .menu-item:hover {
        color: red;
    }
    .dropdown {
        position: relative;
        display: inline-block;
    }
    .dropdown-content {
        display: none;
        position: absolute;
        background-color: rgba(0,0,0,0.8);
        min-width: 160px;
        box-shadow: 0px 8px 16px 0px rgba(0,0,0,0.2);
        z-index: 1;
        left: 50%;
        transform: translateX(-50%);
    }
    .dropdown-content a {
        color: white;
        padding: 12px 16px;
        text-decoration: none;
        display: block;
        text-align: center;
    }
    .dropdown-content a:hover {
        background-color: rgba(255,0,0,0.5);
    }
    .dropdown:hover .dropdown-content {
        display: block;
    }
    .section {
        padding: 20px;
        margin: 20px 0;
        background-color: rgba(0,0,0,0.5);
        color: white;
    }
    .client-logos {
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .client-logos img {
        max-width: 22%;
        height: auto;
    }
    </style>
    """, unsafe_allow_html=True)

    # Modified menu structure with direct model options
    st.markdown("""
    <div class="menu">
        <div class="dropdown">
            <a class="menu-item">Home</a>
            <div class="dropdown-content">
                <a href="#about-us">About Us</a>
                <a href="#our-mission">Our Mission</a>
                <a href="#clients">Clients</a>
            </div>
        </div>
        <a class="menu-item" id="seg-model">Segmentation Model</a>
        <a class="menu-item" id="height-model">Height Segmentation Model</a>
        <div class="dropdown">
            <a class="menu-item">Company</a>
            <div class="dropdown-content">
                <a onclick="navigateTo('company')">Meet the Team</a>
                <a onclick="navigateTo('company')">Careers</a>
            </div>
        </div>
        <a class="menu-item" onclick="navigateTo('contact_us')">Contact Us</a>
    </div>
    """, unsafe_allow_html=True)

    # Add hidden buttons for navigation
    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        if st.button("Segmentation Model", key="seg_model"):
            st.session_state.page = "segmentation_model"
    with col2:
        if st.button("Height Segmentation Model", key="height_model"):
            st.session_state.page = "height_segmentation_model"

    st.markdown("""
    <div id="about-us" class="section">
        <h2>About Us</h2>
        <p>At GeoSeg, we specialize in cutting-edge geospatial technology to optimize wireless communication networks. Our advanced segmentation models harness the power of satellite and LiDAR data to create precise clutter maps, identifying obstacles such as buildings and vegetation that impact radio frequency propagation. 
        By providing detailed environmental insights, we help telecom operators make informed decisions about network design, tower placement, and signal coverage, all while significantly reducing costs associated with traditional clutter data acquisition. GeoSeg is dedicated to delivering innovative, efficient, and affordable solutions for the ever-evolving world of wireless communication.</p>
    </div>
    
    <div id="our-mission" class="section">
        <h2>Our Mission</h2>
        <p>GeoSeg's mission is to revolutionize the way telecom operators approach network planning by providing accurate, cost-effective clutter data through advanced segmentation models. We aim to empower our clients with real-time, actionable insights into their environments, allowing them to optimize signal coverage, improve service quality, and reduce operational costs. 
        By leveraging the latest geospatial technology, we are committed to creating smarter, more reliable communication networks for the future.</p>
    </div>
    
    <div id="clients" class="section">
        <h2>Our Clients</h2>
        <p>Whether you're looking to optimize tower placement, improve coverage, or reduce clutter data costs, we provide the tools and insights needed to make informed, data-driven decisions. 
        Explore our services and see how GeoSeg can support your network optimization needs with precision and reliability.</p>
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
    add_bg_from_local('point-cloud-data.jpg')  # Replace with your image file
    
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