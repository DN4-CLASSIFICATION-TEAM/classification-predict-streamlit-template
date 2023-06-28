if selection == "Welcome":
		
		st.markdown("<h2 style='text-align: center;'>Let's Classify</h2>", unsafe_allow_html=True)
		wel = Image.open("resources/imgs/Welcome.png")
		st.image(wel,use_column_width=True, clamp=False, width = 200, output_format="PNG")
		st.write("Stand Out from the Competition: By using our sentiment analysis app, you'll have an advantage over others because you can respond quickly to public sentiment, address concerns, and align your efforts with what people are saying.")
		st.write("Support for Decision Making: Our app acts as a reliable tool to help you make decisions based on data. You can use the insights to guide your marketing strategies, improve your products, and handle any crises that may arise.")
