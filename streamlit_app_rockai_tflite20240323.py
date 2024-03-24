import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import tempfile
import io

# Load the TFLite model
tflite_model = 'model_fine_20240317.tflite'
interpreter = tf.lite.Interpreter(model_path=tflite_model)
interpreter.allocate_tensors()

# Placeholder for class descriptions (you can fill this from the internet)
class_descriptions = {
    'AMPH': "Amphibolite is a metamorphic rock that contains plagioclase feldspar and amphibole minerals, usually hornblende or actinolite. It is typically dark-colored and dense, with a weakly foliated or flaky structure. It can form from the metamorphism of mafic igneous rocks such as basalt and gabbro, or from the metamorphism of clay-rich sedimentary rocks such as marl or graywacke (Class 0)",
    'BREC': "Breccia is a type of sedimentary rock that consists of large angular fragments of other rocks or minerals, cemented together by a fine-grained material. The fragments are usually gravel-sized or larger and have different shapes and colors depending on the source material. Breccia can form from various processes such as weathering, erosion, volcanic activity, or impact events (Class 1)",
    'BSLT': "Basalt is an igneous rock formed from the rapid cooling of low-viscosity lava rich in magnesium and iron. It is an aphanitic (fine-grained) extrusive rock that often appears dark in color (Class 2)",
    'DACT': "Dacite is a fine-grained igneous rock that is normally light in color. It is often porphyritic. Dacite is found in lava flows, lava domes, dikes, sills, and pyroclastic debris. It is a rock type usually found on continental crust above subduction zones, where a relatively young oceanic plate has melted below (Class 3)",
    'DOLR': "Dolerite, also known as diabase, is an intriguing igneous rock formed from molten magma that cools and solidifies beneath the Earth’s surface. Its composition includes plagioclase feldspar (appearing as white to light gray crystals) and pyroxene minerals (primarily augite, which contributes to its dark color). Minor minerals like olivine, magnetite, and apatite may also be present. Dolerite’s slow crystallization process results in a fine-grained texture and durability. It forms deep within the Earth’s crust or mantle and often occurs as slabs and blocks. The name “dolerite” comes from Greek words meaning “poison stone,” reflecting its dark color and toxic nature. (Class 4)",
    'GNSS': "Gneiss pronounced as 'nais' is a common and widely distributed type of metamorphic rock. It forms through high-temperature and high-pressure metamorphic processes acting on formations composed of igneous or sedimentary rocks. (Class 5)",
    'GRNT': "Granite is a light-colored igneous rock with large grains that can be seen with the eye1. It forms from the slow cooling of magma below the Earths surface. It is mainly composed of quartz and feldspar, with some other minerals. (Class 6)",
    'QTZT': "Quartzite is a nonfoliated metamorphic rock composed almost entirely of quartz. It forms when a quartz-rich sandstone is altered by the heat, pressure, and chemical activity of metamorphism. Metamorphism recrystallizes the sand grains and the silica cement that binds them together. The result is a network of interlocking quartz grains of incredible strength. (Class 7)",
    'SCHT': "Schist is a metamorphic rock that has thin, flat mineral grains arranged in layers or bands. It is mainly composed of platy minerals like mica, feldspar, and quartz. It is easy to split into thin plates or flakes. (Class 8)",
    'SDST': "Sandstone is a sedimentary rock made from sand-sized grains of minerals or fragments of other rocks. It can have different compositions and textures depending on the type and size of the grains. Most sandstone is composed of quartz or feldspar (Class 9)",
    'SHLE': "Shale is a fine-grained sedimentary rock that forms from the compaction of mud consisting of clay and tiny particles of minerals and organic compounds. Shale is the most common sedimentary rock in the Earths crust and it has a property called fissility, which means it can split into thin layers. (Class 10)",
    'SLST': "Siltstone is a sedimentary rock composed mainly of silt-sized particles. It forms where water, wind, or ice deposit silt, and the silt is then compacted and cemented into a rock. Silt accumulates in sedimentary basins throughout the world. It represents a level of current, wave, or wind energy between where sand and mud accumulate. These include fluvial, aeolian, tidal, coastal, lacustrine, deltaic, glacial, paludal, and shelf environments. Sedimentary structures such as layering, cross-bedding, ripple marks, erosional contacts, and fossils provide evidence of these environments. Siltstone is much less common than sandstone and shale. The rock units are usually thinner and less extensive. Only rarely is one notable enough to merit a stratigraphic name. (Class 11)"
}

# Set the page width
st.set_page_config(layout="wide")

# Sidebar
st.sidebar.image("RockAI_logo.PNG", width=200)  # Adjust the width as needed
st.sidebar.image("slb.jpg", width=200)          # Adjust the width as needed
st.sidebar.title("RockAI - Mining Lithology Predictor")
st.sidebar.markdown("---")
st.sidebar.markdown("### What to do?")

# Option buttons
option = st.sidebar.radio("Choose Option", ("Upload Image", "Take Picture"))

# Sidebar about and objective
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
with st.sidebar.expander("Who we are?", expanded=False):
    st.write("This App is called 'RockAI' for 'Mining Lithology Prediction' developed as a capstone project")
with st.sidebar.expander("Project Objective", expanded=False):
    st.write("The objective is to predict the lithology of rocks at the time of drilling with borehole cuttings")

if option == "Upload Image":
    # Upload Image Part
    uploaded_image = st.file_uploader("Choose a borehole core image for Mining lithology prediction...", type=["tif", "tiff", "jpg", "png", "jpeg"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        image = image.resize((224, 224))
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0).astype(np.float32)

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        interpreter.set_tensor(input_details[0]['index'], image_array)
        interpreter.invoke()

        predictions = interpreter.get_tensor(output_details[0]['index'])
        class_names = {'AMPH': 0, 'BREC': 1, 'BSLT': 2, 'DACT': 3, 'DOLR': 4, 'GNSS': 5, 'GRNT': 6, 'QTZT': 7, 'SCHT': 8, 'SDST': 9, 'SHLE': 10, 'SLST': 11}
        predicted_class_index = np.argmax(predictions[0])
        predicted_label_name = list(class_names.keys())[predicted_class_index]

        # Display the predicted label above the image
        st.write(f"Predicted Label: {predicted_label_name}")

        # Create a two-column layout
        col1, col2 = st.columns(2)

        # Display the image in the first column
        col1.image(image, caption=f"Predicted Label: {predicted_label_name}", width=300)

        # Display the class description in the second column
        col2.markdown(f"#### Description for {predicted_label_name}:")
        col2.write(class_descriptions.get(predicted_label_name, "Description not available."))

elif option == "Take Picture":
    # Take picture Part
    take_picture = st.camera_input("Take Picture")

    if take_picture is not None:
        # Decode the image data from the UploadedFile object
        image = Image.open(io.BytesIO(take_picture.read()))

        # Convert the image to grayscale if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Resize the image
        image = image.resize((224, 224))

        # Convert the image to a NumPy array
        image_array = np.array(image)

        # Convert the NumPy array to a compatible data type (uint8)
        image_array = image_array.astype(np.uint8)

        # Process the image for prediction
        image_array = image_array / 255.0
        image_array = np.expand_dims(image_array, axis=0).astype(np.float32)

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        interpreter.set_tensor(input_details[0]['index'], image_array)
        interpreter.invoke()

        predictions = interpreter.get_tensor(output_details[0]['index'])
        class_names = {'AMPH': 0, 'BREC': 1, 'BSLT': 2, 'DACT': 3, 'DOLR': 4, 'GNSS': 5, 'GRNT': 6, 'QTZT': 7, 'SCHT': 8, 'SDST': 9, 'SHLE': 10, 'SLST': 11}
        predicted_class_index = np.argmax(predictions[0])
        predicted_label_name = list(class_names.keys())[predicted_class_index]

        # Display the predicted label above the image
        st.write(f"Predicted Label: {predicted_label_name}")

        # Create a two-column layout
        col1, col2 = st.columns(2)

        # Display the image in the first column
        col1.image(image, caption=f"Predicted Label: {predicted_label_name}", width=300)

        # Display the class description in the second column
        col2.markdown(f"#### Description for {predicted_label_name}:")
        col2.write(class_descriptions.get(predicted_label_name, "Description not available."))
