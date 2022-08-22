from src.classify import classify_news

def main():
    news_articles = [
        """
        Los Angeles | Yoko Ono shocked reporters yesterday when she responded to a 
        question concerning the presidential run of Hillary Clinton and the possibility 
        that she could become the first woman President of the United States in American history.
        The artist and widow of John Lennon, who is in Los Angeles to present a collection of
        cups and saucers she is exhibiting at the Museum of Modern Art, totally took reporters by surprise 
        by admitting she had not only met the former First Lady at various times during a 
        series of protests against the Vietnam War in New York in the 1970s but also knew her “intimately”.
        The celebrity admitted laughingly to having “a fling” with her at the time and acknowledged 
        her election “would be a great advancement for LGBT and Women rights in America” she added.
        """,
        """
        Lisbon: Russian President Vladimir Putin discussed the conflict in Ukraine with his French 
        counterpart Emmanuel Macron by phone on Friday, telling him about Moscow's approach to a potential 
        deal on ceasing hostilities, the Kremlin said.
        "Reacting to concerns expressed by Emmanuel Macron, the Russian president underscored that the 
        Russian armed forces taking part in the special military operation are doing everything possible 
        to preserve the lives of civilians," the Kremlin said in its readout of the call.
        The two also discussed the negotiations between Moscow and Kyiv and Moscow's stance on how a 
        deal could be achieved, it said, without providing more details.
        """,
        """
        United States President, Donald Trump, says he looks forward to working with the administration 
        of Prime Minister, the Most Hon. Andrew Holness, on bilateral and regional issues.
        The President made the comment during a courtesy call paid on him by Jamaica's Ambassador to the 
        United States, Her Excellency Audrey Marks, at the Oval Office in the White House.
        During their discourse, Mr Trump and Ambassador Marks underscored the strong longstanding bond of 
        friendship between the people Jamaica and the United States of America, while noting the island's 
        contribution in many spheres of American life.
        Ms Marks said she looked forward to working with the President in the interest of both countries, 
        and highlighted the synergies of the US-Caribbean nexus cementing the relationship between the nations.
        The Ambassador, in noting the President's agenda of providing more trading opportunities for US 
        companies, called attention to the significant trade surplus the North American country enjoys in 
        the Caribbean.
        She highlighted the fact that the region is the United States seventh largest trading partner, 
        importing non-oil goods and services valued over US$50 billion.
        This, she said, placed the region ahead of U.S. exports to comparatively larger economies such as
         Russia and India combined.
        Ms Marks further noted the mutual benefit and alignment of interest for the continued stability and 
        economic growth of the Caribbean which she described as the United States' third border.
                
        """

    ]
    results = classify_news(news_articles, "cb849745baa6a13b_1647645903")

    if (len(results) == 1) :
        if (results[0] == 0) : print("The news story is most certainly not false information.")
        else : print("The news story is most certainly fraudulent")
    else: 
        for i in range(len(results)):
            out = "Fraudelent" if i == 0 else "Truthy"
            print(out)

    print(f"Model Accuracy : {results[-1]}")


if __name__ == "__main__":
    main()
