"""
Language Detection Dataset Generator
=====================================
Generates a synthetic dataset of sentences in English, Hindi, and Kannada
for use in a Language Detection ML project.

Dataset: ~1050 rows (350 per language)
Columns: Text, Language
Encoding: UTF-8
"""

import pandas as pd
import random
import os

def generate_english_sentences():
    """Generate diverse English sentences across multiple topics."""
    sentences = [
        # Daily Life
        "I went to the grocery store to buy some fresh vegetables and fruits.",
        "She is cooking dinner for the entire family tonight.",
        "We usually go for a morning walk in the nearby park.",
        "The children are playing in the garden after school.",
        "My neighbor just adopted a cute little puppy from the shelter.",
        "I need to pay the electricity bill before the end of this month.",
        "The traffic was really heavy on my way to work today.",
        "She loves reading novels before going to sleep every night.",
        "We celebrated our wedding anniversary at a nice restaurant downtown.",
        "The weather is quite pleasant today compared to yesterday.",
        "I forgot to bring my umbrella and got caught in the rain.",
        "He rides his bicycle to the office every single day.",
        "Our family enjoys watching movies together on weekends.",
        "The new coffee shop near our house serves excellent cappuccino.",
        "She organized her closet and donated old clothes to charity.",
        "I spent the entire afternoon cleaning and organizing my room.",
        "The landlord increased the rent by fifteen percent this year.",
        "We are planning to renovate our kitchen next summer.",
        "He wakes up at five in the morning to practice yoga.",
        "The neighborhood park has beautiful flowering plants and trees.",
        
        # Education
        "The university announced the results of the final examinations last week.",
        "Students are preparing for their competitive entrance examinations.",
        "The professor explained the concept of machine learning in great detail.",
        "Our school organized a science exhibition for all the students.",
        "She received a scholarship to study computer science at a top university.",
        "The library has a vast collection of academic books and journals.",
        "Online education has become increasingly popular in recent years.",
        "The teacher assigned a group project on environmental sustainability.",
        "He completed his doctoral thesis on artificial intelligence applications.",
        "The examination schedule has been released by the university administration.",
        "Many students prefer online courses because of their flexibility.",
        "The school principal addressed the students during the annual function.",
        "She is learning data science through various online platforms.",
        "The mathematics teacher made complex concepts easy to understand.",
        "He scored the highest marks in the physics examination.",
        "The campus placement drive attracted many multinational companies.",
        "Students organized a cultural fest with dance and music performances.",
        "The new academic session will begin from the first week of July.",
        "She taught herself programming by watching tutorial videos online.",
        "The research paper was published in a prestigious international journal.",
        
        # Technology
        "Artificial intelligence is transforming various industries across the world.",
        "The new smartphone features an advanced camera system with optical zoom.",
        "Cloud computing has revolutionized the way businesses store and process data.",
        "Cybersecurity threats are increasing with the growth of digital services.",
        "The software development team released a major update to the application.",
        "Machine learning algorithms can predict customer behavior with high accuracy.",
        "The internet of things connects billions of devices worldwide.",
        "Virtual reality technology is being used in medical training and therapy.",
        "The company developed a new electric vehicle with improved battery life.",
        "Blockchain technology ensures secure and transparent financial transactions.",
        "The latest operating system update includes several new features and fixes.",
        "Data analytics helps companies make informed business decisions.",
        "Robotic process automation reduces manual work in repetitive tasks.",
        "The tech startup raised millions of dollars in its latest funding round.",
        "Quantum computing promises to solve complex problems much faster.",
        "The mobile application has been downloaded over ten million times.",
        "Natural language processing enables machines to understand human language.",
        "Self driving cars are being tested on roads in several countries.",
        "The gaming industry has grown exponentially in the past decade.",
        "Renewable energy technologies are becoming more affordable and efficient.",
        
        # Sports
        "The cricket team won the championship after a thrilling final match.",
        "She broke the national record in the hundred meter sprint event.",
        "The football world cup attracted billions of viewers from around the globe.",
        "The tennis player won three consecutive grand slam titles this year.",
        "Our school basketball team qualified for the state level tournament.",
        "The marathon runner completed the race in under three hours.",
        "He has been training for the upcoming Olympic swimming competition.",
        "The badminton player secured a gold medal at the international championship.",
        "The hockey team displayed excellent teamwork throughout the tournament.",
        "She started playing chess at the age of five and became a champion.",
        "The stadium was packed with enthusiastic fans cheering for their team.",
        "The coach implemented a new training strategy for the upcoming season.",
        "He set a new world record in the long jump competition.",
        "The wrestling championship featured athletes from over fifty countries.",
        "The volleyball team practiced for six hours daily during the training camp.",
        
        # News & Politics
        "The government announced a new policy to boost the economy.",
        "Elections will be held in five states during the next two months.",
        "The prime minister addressed the nation on the independence day celebrations.",
        "A new international trade agreement was signed between the two countries.",
        "The parliament passed a bill to improve healthcare facilities in rural areas.",
        "The finance minister presented the annual budget in the parliament session.",
        "Environmental activists protested against deforestation in the national capital.",
        "The supreme court delivered a landmark judgment on fundamental rights.",
        "A diplomatic summit was held to discuss regional peace and security.",
        "The unemployment rate has decreased significantly in the last quarter.",
        
        # Science
        "Scientists discovered a new species of deep sea fish in the ocean.",
        "The space agency successfully launched a satellite into orbit last night.",
        "Researchers developed a new vaccine that shows promising results in trials.",
        "Climate change is causing significant impacts on global weather patterns.",
        "The discovery of gravitational waves confirmed Einstein's general theory of relativity.",
        "A team of biologists found a new antibiotic compound in soil bacteria.",
        "The telescope captured stunning images of a distant galaxy cluster.",
        "Gene therapy offers new hope for treating previously incurable genetic diseases.",
        "The chemistry experiment demonstrated the principle of chemical equilibrium clearly.",
        "Archaeologists unearthed ancient artifacts dating back thousands of years.",
        
        # Food & Travel
        "The Italian restaurant serves the best handmade pasta in the entire city.",
        "We are planning a road trip to the mountains during the summer vacation.",
        "The street food market offers a variety of delicious and affordable snacks.",
        "She learned to bake bread from scratch during the lockdown period.",
        "The travel guide recommended visiting the historic temples and forts.",
        "The local bakery sells freshly baked croissants every morning.",
        "We explored the beautiful beaches and coral reefs during our island vacation.",
        "The traditional recipe has been passed down through generations in our family.",
        "The national park is home to diverse wildlife and natural landscapes.",
        "He documented his travel experiences in a popular blog online.",
        
        # Weather & Nature
        "Heavy rainfall caused flooding in several low lying areas of the city.",
        "The temperature dropped to minus ten degrees during the winter season.",
        "The monsoon season brings relief from the scorching summer heat.",
        "Beautiful cherry blossoms bloom in the spring across the Japanese countryside.",
        "The drought severely affected agricultural production in the farming regions.",
        "A thick fog covered the highway causing delays in morning traffic.",
        "The sunset over the ocean created a breathtaking display of colors.",
        "Strong winds uprooted several trees and damaged power lines last night.",
        "The snowfall transformed the entire landscape into a winter wonderland.",
        "Birds migrate to warmer regions during the cold winter months.",
        
        # General
        "Music has the power to bring people together regardless of their background.",
        "Reading books regularly helps improve vocabulary and critical thinking skills.",
        "The hospital has state of the art medical equipment and skilled doctors.",
        "Volunteering at the community center gave me a sense of fulfillment.",
        "The art exhibition showcased paintings from local and international artists.",
        "Regular exercise and a balanced diet are essential for good health.",
        "The documentary film highlighted the importance of ocean conservation.",
        "Public transportation needs to be improved to reduce urban congestion.",
        "The museum houses a remarkable collection of historical artifacts and fossils.",
        "Community gardens promote social interaction and provide fresh produce.",
        "The concert was attended by thousands of fans from different cities.",
        "She volunteers at the animal shelter on weekends to help stray dogs.",
        "The new bridge will reduce the commute time between the two cities.",
        "Meditation and mindfulness practices reduce stress and improve mental health.",
        "The film festival screened movies from directors around the world.",
    ]
    return sentences[:350] if len(sentences) >= 350 else sentences * (350 // len(sentences) + 1)


def generate_hindi_sentences():
    """Generate diverse Hindi sentences in Devanagari script."""
    sentences = [
        # Daily Life
        "मैं आज सुबह बाजार गया और ताजी सब्जियाँ खरीदीं।",
        "वह रोज सुबह पार्क में टहलने जाती है।",
        "बच्चे स्कूल से आकर खेल रहे हैं।",
        "आज मौसम बहुत अच्छा है और धूप खिली हुई है।",
        "मेरी माँ ने आज बहुत स्वादिष्ट खाना बनाया है।",
        "हम इस रविवार को पिकनिक पर जा रहे हैं।",
        "उसने नई कार खरीदी है जो बहुत सुंदर दिखती है।",
        "मुझे हर रोज सुबह जल्दी उठना पड़ता है।",
        "बारिश होने की वजह से सड़कों पर पानी भर गया।",
        "दादी माँ ने हमें एक पुरानी कहानी सुनाई।",
        "आज ऑफिस में बहुत काम था और मैं थक गया।",
        "उसने अपने जन्मदिन की पार्टी में सबको बुलाया।",
        "हमारे पड़ोसी बहुत अच्छे और मिलनसार हैं।",
        "मैंने कल रात एक बहुत अच्छी फिल्म देखी।",
        "वह हर शाम को योग और ध्यान करती है।",
        "बाजार में आजकल सब्जियों के दाम बहुत बढ़ गए हैं।",
        "हमने घर में नया फर्नीचर लगवाया है।",
        "मेरे दोस्त ने मुझे एक किताब उपहार में दी।",
        "आज का अखबार पढ़कर मुझे कई नई बातें पता चलीं।",
        "वह ट्रेन से रोजाना दफ्तर जाता है।",
        
        # Education
        "विद्यार्थी परीक्षा की तैयारी में लगे हुए हैं।",
        "हमारे स्कूल में विज्ञान प्रदर्शनी का आयोजन किया गया।",
        "उसने इंजीनियरिंग की पढ़ाई पूरी कर ली है।",
        "शिक्षक ने गणित का कठिन सवाल बहुत आसानी से समझाया।",
        "ऑनलाइन शिक्षा ने पढ़ाई का तरीका बदल दिया है।",
        "वह छात्रवृत्ति पाकर विदेश में पढ़ने गई है।",
        "पुस्तकालय में पढ़ाई के लिए बहुत शांत वातावरण है।",
        "कंप्यूटर विज्ञान आज के समय में बहुत महत्वपूर्ण विषय है।",
        "उसने प्रतियोगी परीक्षा में प्रथम स्थान प्राप्त किया।",
        "शोध कार्य के लिए उसे अंतरराष्ट्रीय पुरस्कार मिला।",
        "नई शिक्षा नीति ने पाठ्यक्रम में कई बदलाव किए हैं।",
        "छात्रों ने वार्षिक उत्सव में सांस्कृतिक कार्यक्रम प्रस्तुत किए।",
        "विश्वविद्यालय ने नए पाठ्यक्रम शुरू करने की घोषणा की।",
        "हिंदी भाषा का साहित्य बहुत समृद्ध और विविध है।",
        "उसने डॉक्टरेट की उपाधि प्राप्त कर ली है।",
        "स्कूल के प्रधानाचार्य ने छात्रों को प्रेरित करने वाला भाषण दिया।",
        "पुस्तक मेले में विभिन्न भाषाओं की किताबें उपलब्ध थीं।",
        "उसने अपना शोध पत्र एक प्रतिष्ठित पत्रिका में प्रकाशित किया।",
        "कक्षा में अध्यापक ने इतिहास की रोचक जानकारी दी।",
        "विद्यालय ने खेल प्रतियोगिता में कई पदक जीते।",
        
        # Technology
        "कृत्रिम बुद्धिमत्ता ने हमारे जीवन को बहुत बदल दिया है।",
        "मोबाइल फोन आज के जीवन का अभिन्न अंग बन गया है।",
        "इंटरनेट ने दुनिया को एक गाँव में बदल दिया है।",
        "सॉफ्टवेयर कंपनी ने नया एप्लिकेशन लॉन्च किया है।",
        "ऑनलाइन खरीदारी ने बाजार की परंपरा को बदल दिया है।",
        "डिजिटल भुगतान से लेनदेन बहुत आसान हो गया है।",
        "साइबर सुरक्षा आज के डिजिटल युग में बेहद जरूरी है।",
        "रोबोटिक्स तकनीक उद्योगों में क्रांतिकारी बदलाव ला रही है।",
        "सोशल मीडिया ने संवाद के तरीके को पूरी तरह बदल दिया।",
        "क्लाउड कंप्यूटिंग से डेटा स्टोरेज आसान हो गया है।",
        "विज्ञान और तकनीक ने चिकित्सा क्षेत्र में अद्भुत प्रगति की है।",
        "नई तकनीक से किसानों की उत्पादकता बढ़ रही है।",
        "ब्लॉकचेन तकनीक वित्तीय लेनदेन को सुरक्षित बनाती है।",
        "वर्चुअल रियलिटी से शिक्षा और प्रशिक्षण में सुधार हो रहा है।",
        "प्रोग्रामिंग सीखना आज के युवाओं के लिए बहुत जरूरी है।",
        "स्मार्ट होम तकनीक ने घर की सुविधाओं को बढ़ा दिया है।",
        "इलेक्ट्रिक वाहनों की मांग तेजी से बढ़ रही है।",
        "डेटा साइंस आज के सबसे लोकप्रिय करियर विकल्पों में है।",
        "ड्रोन तकनीक का उपयोग कृषि में किया जा रहा है।",
        "अंतरिक्ष अनुसंधान में भारत ने बड़ी सफलता हासिल की है।",
        
        # Sports
        "भारतीय क्रिकेट टीम ने विश्व कप जीतकर इतिहास रचा।",
        "उसने ओलंपिक में स्वर्ण पदक जीतकर देश का गौरव बढ़ाया।",
        "कबड्डी विश्व कप में भारत ने शानदार प्रदर्शन किया।",
        "बैडमिंटन खिलाड़ी ने अंतरराष्ट्रीय टूर्नामेंट में जीत हासिल की।",
        "फुटबॉल मैच में दोनों टीमों ने बेहतरीन खेल दिखाया।",
        "हॉकी के खेल में भारतीय टीम ने कमाल का प्रदर्शन किया।",
        "टेनिस खिलाड़ी ने ग्रैंड स्लैम टूर्नामेंट में उपविजेता रही।",
        "एथलीट ने मैराथन दौड़ में नया राष्ट्रीय रिकॉर्ड बनाया।",
        "कुश्ती प्रतियोगिता में उसने स्वर्ण पदक अपने नाम किया।",
        "खेल मंत्रालय ने नई खेल नीति की घोषणा की है।",
        "तैराकी में उसने तीन नए रिकॉर्ड स्थापित किए।",
        "शतरंज खिलाड़ी ने विश्व चैंपियनशिप में दूसरा स्थान प्राप्त किया।",
        "स्टेडियम में हजारों दर्शक मैच देखने के लिए उमड़ पड़े।",
        "प्रशिक्षक ने खिलाड़ियों के लिए नई रणनीति तैयार की है।",
        "खेल अकादमी से कई प्रतिभाशाली खिलाड़ी निकले हैं।",
        
        # News & Politics
        "सरकार ने नई आर्थिक नीति की घोषणा की है।",
        "चुनाव आयोग ने मतदान की तारीख तय कर दी है।",
        "प्रधानमंत्री ने स्वतंत्रता दिवस पर राष्ट्र को संबोधित किया।",
        "संसद में नया विधेयक पारित किया गया है।",
        "वित्त मंत्री ने वार्षिक बजट प्रस्तुत किया है।",
        "न्यायालय ने एक ऐतिहासिक फैसला सुनाया है।",
        "विपक्ष ने सरकार की नीतियों पर सवाल उठाए हैं।",
        "जनता ने शांतिपूर्ण तरीके से अपनी मांगों को रखा।",
        "राज्य सरकार ने किसानों के लिए नई योजना शुरू की।",
        "अंतरराष्ट्रीय सम्मेलन में कई महत्वपूर्ण मुद्दों पर चर्चा हुई।",
        "मुख्यमंत्री ने नई सड़क परियोजना का उद्घाटन किया।",
        "सामाजिक कार्यकर्ताओं ने पर्यावरण संरक्षण की मांग की।",
        "नगर निगम ने शहर की सफाई अभियान शुरू किया है।",
        "राष्ट्रपति ने गणतंत्र दिवस पर देश को बधाई दी।",
        "आर्थिक विकास दर में इस तिमाही में सुधार देखा गया है।",
        
        # Science
        "वैज्ञानिकों ने एक नई दवा की खोज की है।",
        "अंतरिक्ष एजेंसी ने सफलतापूर्वक उपग्रह प्रक्षेपित किया।",
        "जलवायु परिवर्तन पूरी दुनिया के लिए एक बड़ी चुनौती है।",
        "शोधकर्ताओं ने कैंसर के इलाज में नई प्रगति की है।",
        "सौर ऊर्जा स्वच्छ और नवीकरणीय ऊर्जा का स्रोत है।",
        "मंगल ग्रह पर पानी के अवशेष मिलने की संभावना है।",
        "जैव विविधता का संरक्षण हमारे पारिस्थितिकी तंत्र के लिए जरूरी है।",
        "भौतिक विज्ञान में नई खोजें रोमांचक हैं।",
        "रसायन विज्ञान के प्रयोग में सावधानी बरतनी चाहिए।",
        "चंद्रमा पर भारत का मिशन सफल रहा।",
        
        # Food & Culture
        "भारतीय व्यंजन अपने मसालों और स्वाद के लिए प्रसिद्ध हैं।",
        "दिवाली भारत का सबसे बड़ा और लोकप्रिय त्योहार है।",
        "राजस्थान की संस्कृति और कला विश्व प्रसिद्ध है।",
        "चाय भारत में सबसे लोकप्रिय पेय पदार्थ है।",
        "होली रंगों का त्योहार है जो बसंत ऋतु में मनाया जाता है।",
        "भारतीय शास्त्रीय संगीत की परंपरा बहुत पुरानी है।",
        "ताजमहल विश्व के सात अजूबों में से एक है।",
        "योग और आयुर्वेद भारत की प्राचीन विधाएँ हैं।",
        "गंगा नदी भारत की सबसे पवित्र नदी मानी जाती है।",
        "भारतीय फिल्म उद्योग विश्व का सबसे बड़ा फिल्म उद्योग है।",
        
        # Weather & Travel
        "मानसून के मौसम में खूब बारिश होती है।",
        "सर्दियों में पहाड़ों पर बर्फबारी होती है।",
        "गर्मियों में तापमान पैंतालीस डिग्री तक पहुँच जाता है।",
        "हिमालय की चोटियाँ बहुत सुंदर और भव्य दिखती हैं।",
        "केरल को भगवान का अपना देश कहा जाता है।",
        "लद्दाख की प्राकृतिक सुंदरता अद्भुत है।",
        "वाराणसी दुनिया के सबसे प्राचीन शहरों में से एक है।",
        "गोवा के समुद्र तट पर्यटकों में बहुत लोकप्रिय हैं।",
        "जयपुर को गुलाबी शहर के नाम से जाना जाता है।",
        "दार्जिलिंग की चाय की बागान देखने लायक हैं।",
        
        # Additional General
        "किताबें पढ़ना ज्ञान बढ़ाने का सबसे अच्छा तरीका है।",
        "संगीत सुनने से मन को शांति मिलती है।",
        "परिवार का साथ जीवन की सबसे बड़ी खुशी है।",
        "स्वास्थ्य ही सबसे बड़ा धन है।",
        "समय का सदुपयोग करना सफलता की कुंजी है।",
        "पर्यावरण की रक्षा करना हम सबकी जिम्मेदारी है।",
        "शिक्षा समाज के विकास की नींव है।",
        "भ्रष्टाचार देश के विकास में सबसे बड़ी बाधा है।",
        "महिला सशक्तिकरण आज के समय की सबसे बड़ी जरूरत है।",
        "सभी धर्म प्रेम और शांति का संदेश देते हैं।",
        "बचपन जीवन का सबसे खूबसूरत समय होता है।",
        "मेहनत और लगन से हर लक्ष्य हासिल किया जा सकता है।",
        "देश की प्रगति में युवाओं की भूमिका महत्वपूर्ण है।",
        "अच्छे संस्कार बच्चों के भविष्य को उज्ज्वल बनाते हैं।",
        "प्रकृति की सुंदरता हमें विस्मित कर देती है।",
        "स्वच्छता अभियान ने लोगों में जागरूकता फैलाई है।",
        "ग्रामीण क्षेत्रों में विकास की गति तेज होनी चाहिए।",
        "भारतीय संस्कृति विविधता में एकता का प्रतीक है।",
        "सही आहार और व्यायाम से शरीर स्वस्थ रहता है।",
        "बड़ों का आशीर्वाद जीवन में बहुत महत्वपूर्ण होता है।",
    ]
    return sentences[:350] if len(sentences) >= 350 else sentences * (350 // len(sentences) + 1)


def generate_kannada_sentences():
    """Generate diverse Kannada sentences in Kannada script."""
    sentences = [
        # Daily Life
        "ನಾನು ಇಂದು ಬೆಳಿಗ್ಗೆ ಮಾರುಕಟ್ಟೆಗೆ ಹೋಗಿ ತರಕಾರಿ ತಂದೆ.",
        "ಅವಳು ಪ್ರತಿದಿನ ಬೆಳಿಗ್ಗೆ ಉದ್ಯಾನವನದಲ್ಲಿ ನಡೆಯುತ್ತಾಳೆ.",
        "ಮಕ್ಕಳು ಶಾಲೆಯಿಂದ ಬಂದ ನಂತರ ಆಟವಾಡುತ್ತಿದ್ದಾರೆ.",
        "ಇಂದು ಹವಾಮಾನ ತುಂಬಾ ಚೆನ್ನಾಗಿದೆ ಮತ್ತು ಬಿಸಿಲು ಚೆನ್ನಾಗಿದೆ.",
        "ನನ್ನ ಅಮ್ಮ ಇಂದು ರುಚಿಯಾದ ಊಟ ಮಾಡಿದ್ದಾರೆ.",
        "ನಾವು ಈ ಭಾನುವಾರ ಪಿಕ್ನಿಕ್ ಗೆ ಹೋಗುತ್ತಿದ್ದೇವೆ.",
        "ಅವನು ಹೊಸ ಕಾರು ಖರೀದಿಸಿದ್ದಾನೆ ಅದು ಬಹಳ ಸುಂದರವಾಗಿದೆ.",
        "ನನಗೆ ಪ್ರತಿದಿನ ಬೆಳಿಗ್ಗೆ ಬೇಗ ಏಳಬೇಕಾಗುತ್ತದೆ.",
        "ಮಳೆ ಬಂದ ಕಾರಣ ರಸ್ತೆಗಳಲ್ಲಿ ನೀರು ತುಂಬಿದೆ.",
        "ಅಜ್ಜಿ ನಮಗೆ ಒಂದು ಹಳೆಯ ಕಥೆ ಹೇಳಿದರು.",
        "ಇಂದು ಕಚೇರಿಯಲ್ಲಿ ಬಹಳ ಕೆಲಸ ಇತ್ತು ನಾನು ದಣಿದಿದ್ದೇನೆ.",
        "ಅವಳು ತನ್ನ ಹುಟ್ಟುಹಬ್ಬದ ಪಾರ್ಟಿಗೆ ಎಲ್ಲರನ್ನೂ ಕರೆದಳು.",
        "ನಮ್ಮ ನೆರೆಹೊರೆಯವರು ತುಂಬಾ ಒಳ್ಳೆಯವರು.",
        "ನಾನು ನಿನ್ನೆ ರಾತ್ರಿ ಒಂದು ಒಳ್ಳೆಯ ಚಲನಚಿತ್ರ ನೋಡಿದೆ.",
        "ಅವಳು ಪ್ರತಿ ಸಂಜೆ ಯೋಗ ಮತ್ತು ಧ್ಯಾನ ಮಾಡುತ್ತಾಳೆ.",
        "ಮಾರುಕಟ್ಟೆಯಲ್ಲಿ ತರಕಾರಿ ಬೆಲೆ ಬಹಳ ಹೆಚ್ಚಾಗಿದೆ.",
        "ನಾವು ಮನೆಯಲ್ಲಿ ಹೊಸ ಪೀಠೋಪಕರಣ ಹಾಕಿಸಿದ್ದೇವೆ.",
        "ನನ್ನ ಸ್ನೇಹಿತ ನನಗೆ ಒಂದು ಪುಸ್ತಕ ಉಡುಗೊರೆಯಾಗಿ ಕೊಟ್ಟ.",
        "ಇಂದಿನ ಪತ್ರಿಕೆ ಓದಿ ಹಲವು ಹೊಸ ವಿಷಯಗಳು ತಿಳಿದವು.",
        "ಅವನು ರೈಲಿನಲ್ಲಿ ಪ್ರತಿದಿನ ಕಚೇರಿಗೆ ಹೋಗುತ್ತಾನೆ.",
        
        # Education
        "ವಿದ್ಯಾರ್ಥಿಗಳು ಪರೀಕ್ಷೆಯ ತಯಾರಿಯಲ್ಲಿ ನಿರತರಾಗಿದ್ದಾರೆ.",
        "ನಮ್ಮ ಶಾಲೆಯಲ್ಲಿ ವಿಜ್ಞಾನ ಪ್ರದರ್ಶನ ಆಯೋಜಿಸಲಾಗಿದೆ.",
        "ಅವಳು ಎಂಜಿನಿಯರಿಂಗ್ ಪದವಿ ಪೂರ್ಣಗೊಳಿಸಿದ್ದಾಳೆ.",
        "ಶಿಕ್ಷಕರು ಗಣಿತದ ಕಷ್ಟದ ಪ್ರಶ್ನೆಯನ್ನು ಸುಲಭವಾಗಿ ವಿವರಿಸಿದರು.",
        "ಆನ್‌ಲೈನ್ ಶಿಕ್ಷಣ ಕಲಿಕೆಯ ವಿಧಾನವನ್ನು ಬದಲಾಯಿಸಿದೆ.",
        "ಅವಳು ವಿದ್ಯಾರ್ಥಿವೇತನ ಪಡೆದು ವಿದೇಶದಲ್ಲಿ ಓದಲು ಹೋಗಿದ್ದಾಳೆ.",
        "ಗ್ರಂಥಾಲಯದಲ್ಲಿ ಅಧ್ಯಯನಕ್ಕೆ ಶಾಂತ ವಾತಾವರಣವಿದೆ.",
        "ಕಂಪ್ಯೂಟರ್ ವಿಜ್ಞಾನ ಇಂದಿನ ಅತ್ಯಂತ ಮುಖ್ಯ ವಿಷಯವಾಗಿದೆ.",
        "ಅವನು ಸ್ಪರ್ಧಾತ್ಮಕ ಪರೀಕ್ಷೆಯಲ್ಲಿ ಪ್ರಥಮ ಸ್ಥಾನ ಪಡೆದಿದ್ದಾನೆ.",
        "ಸಂಶೋಧನಾ ಕಾರ್ಯಕ್ಕಾಗಿ ಅಂತಾರಾಷ್ಟ್ರೀಯ ಪ್ರಶಸ್ತಿ ಸಿಕ್ಕಿದೆ.",
        "ಹೊಸ ಶಿಕ್ಷಣ ನೀತಿ ಪಠ್ಯಕ್ರಮದಲ್ಲಿ ಬದಲಾವಣೆ ತಂದಿದೆ.",
        "ವಿಶ್ವವಿದ್ಯಾಲಯ ಹೊಸ ಕೋರ್ಸ್‌ಗಳನ್ನು ಪ್ರಾರಂಭಿಸುವ ಘೋಷಣೆ ಮಾಡಿದೆ.",
        "ಕನ್ನಡ ಸಾಹಿತ್ಯ ಬಹಳ ಶ್ರೀಮಂತ ಮತ್ತು ವೈವಿಧ್ಯಮಯವಾಗಿದೆ.",
        "ಶಾಲಾ ಮುಖ್ಯ ಶಿಕ್ಷಕರು ವಿದ್ಯಾರ್ಥಿಗಳಿಗೆ ಸ್ಫೂರ್ತಿದಾಯಕ ಭಾಷಣ ಮಾಡಿದರು.",
        "ಪುಸ್ತಕ ಮೇಳದಲ್ಲಿ ವಿವಿಧ ಭಾಷೆಗಳ ಪುಸ್ತಕಗಳು ಲಭ್ಯವಿದ್ದವು.",
        "ಅವಳು ತನ್ನ ಸಂಶೋಧನಾ ಪ್ರಬಂಧವನ್ನು ಪ್ರತಿಷ್ಠಿತ ಪತ್ರಿಕೆಯಲ್ಲಿ ಪ್ರಕಟಿಸಿದಳು.",
        "ತರಗತಿಯಲ್ಲಿ ಶಿಕ್ಷಕರು ಇತಿಹಾಸದ ಆಸಕ್ತಿಕರ ಮಾಹಿತಿ ನೀಡಿದರು.",
        "ಶಾಲೆ ಕ್ರೀಡಾ ಸ್ಪರ್ಧೆಯಲ್ಲಿ ಅನೇಕ ಪದಕಗಳನ್ನು ಗೆದ್ದಿದೆ.",
        "ವಿದ್ಯಾರ್ಥಿಗಳು ವಾರ್ಷಿಕ ಉತ್ಸವದಲ್ಲಿ ಸಾಂಸ್ಕೃತಿಕ ಕಾರ್ಯಕ್ರಮ ಮಂಡಿಸಿದರು.",
        "ಅವನು ಡಾಕ್ಟರೇಟ್ ಪದವಿ ಪಡೆದಿದ್ದಾನೆ.",
        
        # Technology
        "ಕೃತಕ ಬುದ್ಧಿಮತ್ತೆ ನಮ್ಮ ಜೀವನವನ್ನು ಬಹಳ ಬದಲಾಯಿಸಿದೆ.",
        "ಮೊಬೈಲ್ ಫೋನ್ ಇಂದಿನ ಜೀವನದ ಅವಿಭಾಜ್ಯ ಅಂಗವಾಗಿದೆ.",
        "ಅಂತರ್ಜಾಲ ಜಗತ್ತನ್ನು ಒಂದು ಹಳ್ಳಿಯನ್ನಾಗಿ ಮಾಡಿದೆ.",
        "ಸಾಫ್ಟ್‌ವೇರ್ ಕಂಪನಿ ಹೊಸ ಅಪ್ಲಿಕೇಶನ್ ಬಿಡುಗಡೆ ಮಾಡಿದೆ.",
        "ಆನ್‌ಲೈನ್ ಖರೀದಿ ಮಾರುಕಟ್ಟೆಯ ಸಂಪ್ರದಾಯವನ್ನು ಬದಲಾಯಿಸಿದೆ.",
        "ಡಿಜಿಟಲ್ ಪಾವತಿ ವ್ಯವಹಾರಗಳನ್ನು ಸುಲಭಗೊಳಿಸಿದೆ.",
        "ಸೈಬರ್ ಭದ್ರತೆ ಡಿಜಿಟಲ್ ಯುಗದಲ್ಲಿ ಅತ್ಯಂತ ಅವಶ್ಯಕವಾಗಿದೆ.",
        "ರೋಬೋಟಿಕ್ಸ್ ತಂತ್ರಜ್ಞಾನ ಉದ್ಯಮಗಳಲ್ಲಿ ಕ್ರಾಂತಿಕಾರಿ ಬದಲಾವಣೆ ತರುತ್ತಿದೆ.",
        "ಸಾಮಾಜಿಕ ಮಾಧ್ಯಮ ಸಂವಹನ ವಿಧಾನವನ್ನು ಸಂಪೂರ್ಣವಾಗಿ ಬದಲಾಯಿಸಿದೆ.",
        "ಕ್ಲೌಡ್ ಕಂಪ್ಯೂಟಿಂಗ್ ಡೇಟಾ ಸಂಗ್ರಹಣೆಯನ್ನು ಸುಲಭಗೊಳಿಸಿದೆ.",
        "ವಿಜ್ಞಾನ ಮತ್ತು ತಂತ್ರಜ್ಞಾನ ವೈದ್ಯಕೀಯ ಕ್ಷೇತ್ರದಲ್ಲಿ ಅದ್ಭುತ ಪ್ರಗತಿ ಸಾಧಿಸಿದೆ.",
        "ಹೊಸ ತಂತ್ರಜ್ಞಾನದಿಂದ ರೈತರ ಉತ್ಪಾದಕತೆ ಹೆಚ್ಚುತ್ತಿದೆ.",
        "ಬ್ಲಾಕ್‌ಚೈನ್ ತಂತ್ರಜ್ಞಾನ ಹಣಕಾಸು ವ್ಯವಹಾರಗಳನ್ನು ಸುರಕ್ಷಿತಗೊಳಿಸುತ್ತದೆ.",
        "ವರ್ಚುವಲ್ ರಿಯಾಲಿಟಿ ಶಿಕ್ಷಣ ಮತ್ತು ತರಬೇತಿಯನ್ನು ಸುಧಾರಿಸುತ್ತಿದೆ.",
        "ಪ್ರೋಗ್ರಾಮಿಂಗ್ ಕಲಿಯುವುದು ಯುವಜನರಿಗೆ ಅತ್ಯಗತ್ಯವಾಗಿದೆ.",
        "ಸ್ಮಾರ್ಟ್ ಹೋಮ್ ತಂತ್ರಜ್ಞಾನ ಮನೆಯ ಸೌಕರ್ಯಗಳನ್ನು ಹೆಚ್ಚಿಸಿದೆ.",
        "ವಿದ್ಯುತ್ ವಾಹನಗಳ ಬೇಡಿಕೆ ವೇಗವಾಗಿ ಹೆಚ್ಚುತ್ತಿದೆ.",
        "ಡೇಟಾ ಸೈನ್ಸ್ ಇಂದಿನ ಅತ್ಯಂತ ಜನಪ್ರಿಯ ವೃತ್ತಿ ಆಯ್ಕೆಗಳಲ್ಲಿ ಒಂದಾಗಿದೆ.",
        "ಡ್ರೋನ್ ತಂತ್ರಜ್ಞಾನವನ್ನು ಕೃಷಿಯಲ್ಲಿ ಬಳಸಲಾಗುತ್ತಿದೆ.",
        "ಬಾಹ್ಯಾಕಾಶ ಸಂಶೋಧನೆಯಲ್ಲಿ ಭಾರತ ದೊಡ್ಡ ಯಶಸ್ಸು ಸಾಧಿಸಿದೆ.",
        
        # Sports
        "ಭಾರತೀಯ ಕ್ರಿಕೆಟ್ ತಂಡ ವಿಶ್ವಕಪ್ ಗೆದ್ದು ಇತಿಹಾಸ ರಚಿಸಿತು.",
        "ಅವಳು ಒಲಿಂಪಿಕ್ಸ್‌ನಲ್ಲಿ ಚಿನ್ನದ ಪದಕ ಗೆದ್ದು ದೇಶದ ಹೆಮ್ಮೆ ಹೆಚ್ಚಿಸಿದಳು.",
        "ಕಬಡ್ಡಿ ವಿಶ್ವಕಪ್‌ನಲ್ಲಿ ಭಾರತ ಅದ್ಭುತ ಪ್ರದರ್ಶನ ನೀಡಿತು.",
        "ಬ್ಯಾಡ್ಮಿಂಟನ್ ಆಟಗಾರ ಅಂತಾರಾಷ್ಟ್ರೀಯ ಪಂದ್ಯಾವಳಿಯಲ್ಲಿ ಗೆಲುವು ಸಾಧಿಸಿದ.",
        "ಫುಟ್ಬಾಲ್ ಪಂದ್ಯದಲ್ಲಿ ಎರಡೂ ತಂಡಗಳು ಅತ್ಯುತ್ತಮ ಆಟ ತೋರಿಸಿದವು.",
        "ಹಾಕಿ ಆಟದಲ್ಲಿ ಭಾರತೀಯ ತಂಡ ಅದ್ಭುತ ಪ್ರದರ್ಶನ ನೀಡಿತು.",
        "ಟೆನ್ನಿಸ್ ಆಟಗಾರ ಗ್ರ್ಯಾಂಡ್ ಸ್ಲ್ಯಾಮ್ ಪಂದ್ಯಾವಳಿಯಲ್ಲಿ ಉಪವಿಜೇತರಾದರು.",
        "ಕ್ರೀಡಾಪಟು ಮ್ಯಾರಥಾನ್ ಓಟದಲ್ಲಿ ಹೊಸ ರಾಷ್ಟ್ರೀಯ ದಾಖಲೆ ಮಾಡಿದರು.",
        "ಕುಸ್ತಿ ಸ್ಪರ್ಧೆಯಲ್ಲಿ ಅವನು ಚಿನ್ನದ ಪದಕ ಗೆದ್ದನು.",
        "ಕ್ರೀಡಾ ಸಚಿವಾಲಯ ಹೊಸ ಕ್ರೀಡಾ ನೀತಿ ಘೋಷಿಸಿದೆ.",
        "ಈಜುವಿಕೆಯಲ್ಲಿ ಅವಳು ಮೂರು ಹೊಸ ದಾಖಲೆ ಸ್ಥಾಪಿಸಿದಳು.",
        "ಚೆಸ್ ಆಟಗಾರ ವಿಶ್ವ ಚಾಂಪಿಯನ್‌ಶಿಪ್‌ನಲ್ಲಿ ಎರಡನೇ ಸ್ಥಾನ ಪಡೆದರು.",
        "ಕ್ರೀಡಾಂಗಣದಲ್ಲಿ ಸಾವಿರಾರು ಪ್ರೇಕ್ಷಕರು ಪಂದ್ಯ ನೋಡಲು ಬಂದರು.",
        "ತರಬೇತುದಾರರು ಆಟಗಾರರಿಗೆ ಹೊಸ ತಂತ್ರ ತಯಾರಿಸಿದ್ದಾರೆ.",
        "ಕ್ರೀಡಾ ಅಕಾಡೆಮಿಯಿಂದ ಅನೇಕ ಪ್ರತಿಭಾವಂತ ಆಟಗಾರರು ಬಂದಿದ್ದಾರೆ.",
        
        # News & Politics
        "ಸರ್ಕಾರ ಹೊಸ ಆರ್ಥಿಕ ನೀತಿಯನ್ನು ಘೋಷಿಸಿದೆ.",
        "ಚುನಾವಣಾ ಆಯೋಗ ಮತದಾನದ ದಿನಾಂಕ ನಿಗದಿ ಮಾಡಿದೆ.",
        "ಪ್ರಧಾನಮಂತ್ರಿ ಸ್ವಾತಂತ್ರ್ಯ ದಿನದಂದು ರಾಷ್ಟ್ರವನ್ನು ಉದ್ದೇಶಿಸಿ ಮಾತನಾಡಿದರು.",
        "ಸಂಸತ್ತಿನಲ್ಲಿ ಹೊಸ ಮಸೂದೆ ಅಂಗೀಕಾರವಾಗಿದೆ.",
        "ಹಣಕಾಸು ಸಚಿವರು ವಾರ್ಷಿಕ ಬಜೆಟ್ ಮಂಡಿಸಿದ್ದಾರೆ.",
        "ನ್ಯಾಯಾಲಯ ಒಂದು ಐತಿಹಾಸಿಕ ತೀರ್ಪು ನೀಡಿದೆ.",
        "ವಿಪಕ್ಷ ಸರ್ಕಾಡ ನೀತಿಗಳ ಬಗ್ಗೆ ಪ್ರಶ್ನೆ ಎತ್ತಿದೆ.",
        "ಜನರು ಶಾಂತಿಯುತವಾಗಿ ತಮ್ಮ ಬೇಡಿಕೆಗಳನ್ನು ಮಂಡಿಸಿದರು.",
        "ರಾಜ್ಯ ಸರ್ಕಾರ ರೈತರಿಗೆ ಹೊಸ ಯೋಜನೆ ಆರಂಭಿಸಿದೆ.",
        "ಅಂತಾರಾಷ್ಟ್ರೀಯ ಸಮ್ಮೇಳನದಲ್ಲಿ ಹಲವು ಮಹತ್ವದ ವಿಷಯಗಳ ಚರ್ಚೆ ನಡೆಯಿತು.",
        "ಮುಖ್ಯಮಂತ್ರಿ ಹೊಸ ರಸ್ತೆ ಯೋಜನೆಯನ್ನು ಉದ್ಘಾಟಿಸಿದರು.",
        "ಸಾಮಾಜಿಕ ಕಾರ್ಯಕರ್ತರು ಪರಿಸರ ಸಂರಕ್ಷಣೆಯ ಬೇಡಿಕೆ ಇಟ್ಟರು.",
        "ನಗರಪಾಲಿಕೆ ನಗರದ ಸ್ವಚ್ಛತಾ ಅಭಿಯಾನ ಆರಂಭಿಸಿದೆ.",
        "ರಾಷ್ಟ್ರಪತಿ ಗಣರಾಜ್ಯೋತ್ಸವದಂದು ದೇಶಕ್ಕೆ ಶುಭಾಶಯ ತಿಳಿಸಿದರು.",
        "ಆರ್ಥಿಕ ಅಭಿವೃದ್ಧಿ ದರದಲ್ಲಿ ಈ ತ್ರೈಮಾಸಿಕದಲ್ಲಿ ಸುಧಾರಣೆ ಕಂಡಿದೆ.",
        
        # Science
        "ವಿಜ್ಞಾನಿಗಳು ಹೊಸ ಔಷಧಿಯನ್ನು ಕಂಡುಹಿಡಿದಿದ್ದಾರೆ.",
        "ಬಾಹ್ಯಾಕಾಶ ಸಂಸ್ಥೆ ಯಶಸ್ವಿಯಾಗಿ ಉಪಗ್ರಹ ಉಡಾವಣೆ ಮಾಡಿದೆ.",
        "ಹವಾಮಾನ ಬದಲಾವಣೆ ಇಡೀ ಜಗತ್ತಿಗೆ ದೊಡ್ಡ ಸವಾಲಾಗಿದೆ.",
        "ಸಂಶೋಧಕರು ಕ್ಯಾನ್ಸರ್ ಚಿಕಿತ್ಸೆಯಲ್ಲಿ ಹೊಸ ಪ್ರಗತಿ ಸಾಧಿಸಿದ್ದಾರೆ.",
        "ಸೌರ ಶಕ್ತಿ ಶುದ್ಧ ಮತ್ತು ನವೀಕರಿಸಬಹುದಾದ ಶಕ್ತಿಯ ಮೂಲವಾಗಿದೆ.",
        "ಮಂಗಳ ಗ್ರಹದಲ್ಲಿ ನೀರಿನ ಅವಶೇಷ ಸಿಗುವ ಸಾಧ್ಯತೆ ಇದೆ.",
        "ಜೀವವೈವಿಧ್ಯ ಸಂರಕ್ಷಣೆ ನಮ್ಮ ಪರಿಸರ ವ್ಯವಸ್ಥೆಗೆ ಅಗತ್ಯವಾಗಿದೆ.",
        "ಭೌತಶಾಸ್ತ್ರದಲ್ಲಿ ಹೊಸ ಆವಿಷ್ಕಾರಗಳು ರೋಮಾಂಚಕವಾಗಿವೆ.",
        "ರಸಾಯನಶಾಸ್ತ್ರ ಪ್ರಯೋಗದಲ್ಲಿ ಜಾಗರೂಕತೆ ವಹಿಸಬೇಕು.",
        "ಚಂದ್ರನ ಮೇಲಿನ ಭಾರತದ ಯೋಜನೆ ಯಶಸ್ವಿಯಾಯಿತು.",
        
        # Culture & Food
        "ಕರ್ನಾಟಕದ ಸಂಸ್ಕೃತಿ ಮತ್ತು ಪರಂಪರೆ ಅತ್ಯಂತ ಶ್ರೀಮಂತವಾಗಿದೆ.",
        "ಮೈಸೂರು ದಸರಾ ಕರ್ನಾಟಕದ ಅತ್ಯಂತ ಪ್ರಸಿದ್ಧ ಹಬ್ಬವಾಗಿದೆ.",
        "ಬಿಸಿ ಬೇಳೆ ಬಾತ್ ಕರ್ನಾಟಕದ ಜನಪ್ರಿಯ ಖಾದ್ಯವಾಗಿದೆ.",
        "ಕನ್ನಡ ಚಲನಚಿತ್ರ ಉದ್ಯಮ ಬಹಳ ಬೆಳೆದಿದೆ.",
        "ಹಂಪಿ ವಿಶ್ವ ಪರಂಪರೆಯ ತಾಣವಾಗಿದೆ.",
        "ಯಕ್ಷಗಾನ ಕರ್ನಾಟಕದ ಪ್ರಸಿದ್ಧ ಜಾನಪದ ಕಲೆಯಾಗಿದೆ.",
        "ಕೂರ್ಗ್ ಕಾಫಿ ತೋಟಗಳಿಗೆ ಪ್ರಸಿದ್ಧವಾಗಿದೆ.",
        "ಬೆಂಗಳೂರು ಭಾರತದ ಸಿಲಿಕಾನ್ ವ್ಯಾಲಿ ಎಂದು ಕರೆಯಲ್ಪಡುತ್ತದೆ.",
        "ರಾಗಿ ಮುದ್ದೆ ಕರ್ನಾಟಕದ ಸಾಂಪ್ರದಾಯಿಕ ಆಹಾರವಾಗಿದೆ.",
        "ಗೋಲ್ ಗುಂಬಜ್ ವಿಶ್ವದ ಎರಡನೇ ಅತಿ ದೊಡ್ಡ ಗುಮ್ಮಟವಾಗಿದೆ.",
        
        # Weather & General
        "ಮಳೆಗಾಲದಲ್ಲಿ ಬಹಳ ಮಳೆ ಬರುತ್ತದೆ.",
        "ಚಳಿಗಾಲದಲ್ಲಿ ಬೆಟ್ಟಗಳ ಮೇಲೆ ಹಿಮ ಬೀಳುತ್ತದೆ.",
        "ಬೇಸಿಗೆಯಲ್ಲಿ ತಾಪಮಾನ ನಲವತ್ತೈದು ಡಿಗ್ರಿ ತಲುಪುತ್ತದೆ.",
        "ಪುಸ್ತಕ ಓದುವುದು ಜ್ಞಾನ ಹೆಚ್ಚಿಸಲು ಉತ್ತಮ ಮಾರ್ಗವಾಗಿದೆ.",
        "ಸಂಗೀತ ಕೇಳುವುದು ಮನಸ್ಸಿಗೆ ಶಾಂತಿ ನೀಡುತ್ತದೆ.",
        "ಕುಟುಂಬದ ಸಹವಾಸ ಜೀವನದ ಅತ್ಯಂತ ದೊಡ್ಡ ಸಂತೋಷವಾಗಿದೆ.",
        "ಆರೋಗ್ಯವೇ ಅತ್ಯಂತ ದೊಡ್ಡ ಸಂಪತ್ತು.",
        "ಸಮಯದ ಸದ್ಬಳಕೆ ಯಶಸ್ಸಿನ ಕೀಲಿಯಾಗಿದೆ.",
        "ಪರಿಸರ ರಕ್ಷಣೆ ನಮ್ಮೆಲ್ಲರ ಹೊಣೆಯಾಗಿದೆ.",
        "ಶಿಕ್ಷಣ ಸಮಾಜದ ಅಭಿವೃದ್ಧಿಯ ಅಡಿಪಾಯವಾಗಿದೆ.",
        "ಮಹಿಳಾ ಸಬಲೀಕರಣ ಇಂದಿನ ಅತ್ಯಂತ ದೊಡ್ಡ ಅಗತ್ಯವಾಗಿದೆ.",
        "ಎಲ್ಲಾ ಧರ್ಮಗಳು ಪ್ರೀತಿ ಮತ್ತು ಶಾಂತಿಯ ಸಂದೇಶ ನೀಡುತ್ತವೆ.",
        "ಬಾಲ್ಯ ಜೀವನದ ಅತ್ಯಂತ ಸುಂದರ ಸಮಯವಾಗಿದೆ.",
        "ಶ್ರಮ ಮತ್ತು ಶ್ರದ್ಧೆಯಿಂದ ಯಾವುದೇ ಗುರಿ ಸಾಧಿಸಬಹುದು.",
        "ದೇಶದ ಪ್ರಗತಿಯಲ್ಲಿ ಯುವಜನರ ಪಾತ್ರ ಮಹತ್ವದ್ದಾಗಿದೆ.",
        "ಒಳ್ಳೆಯ ಸಂಸ್ಕಾರಗಳು ಮಕ್ಕಳ ಭವಿಷ್ಯವನ್ನು ಉಜ್ವಲಗೊಳಿಸುತ್ತವೆ.",
        "ಪ್ರಕೃತಿಯ ಸೌಂದರ್ಯ ನಮ್ಮನ್ನು ಬೆರಗುಗೊಳಿಸುತ್ತದೆ.",
        "ಸ್ವಚ್ಛತಾ ಅಭಿಯಾನ ಜನರಲ್ಲಿ ಜಾಗೃತಿ ಮೂಡಿಸಿದೆ.",
        "ಗ್ರಾಮೀಣ ಪ್ರದೇಶಗಳಲ್ಲಿ ಅಭಿವೃದ್ಧಿ ವೇಗ ಹೆಚ್ಚಬೇಕಿದೆ.",
        "ಕನ್ನಡ ನಾಡಿನ ಸಂಸ್ಕೃತಿ ವೈವಿಧ್ಯಮಯ ಮತ್ತು ಶ್ರೀಮಂತವಾಗಿದೆ.",
    ]
    return sentences[:350] if len(sentences) >= 350 else sentences * (350 // len(sentences) + 1)


def create_dataset():
    """Generate and save the language detection dataset."""
    print("=" * 60)
    print("Language Detection Dataset Generator")
    print("=" * 60)
    
    # Create data directory
    os.makedirs("data", exist_ok=True)
    
    # Generate sentences
    print("\n[1/4] Generating English sentences...")
    english = generate_english_sentences()[:350]
    print(f"      Generated {len(english)} English sentences")
    
    print("[2/4] Generating Hindi sentences...")
    hindi = generate_hindi_sentences()[:350]
    print(f"      Generated {len(hindi)} Hindi sentences")
    
    print("[3/4] Generating Kannada sentences...")
    kannada = generate_kannada_sentences()[:350]
    print(f"      Generated {len(kannada)} Kannada sentences")
    
    # Combine into dataframe
    print("[4/4] Creating dataset...")
    data = []
    for text in english:
        data.append({"Text": text, "Language": "English"})
    for text in hindi:
        data.append({"Text": text, "Language": "Hindi"})
    for text in kannada:
        data.append({"Text": text, "Language": "Kannada"})
    
    df = pd.DataFrame(data)
    
    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save
    output_path = os.path.join("data", "language_dataset.csv")
    df.to_csv(output_path, index=False, encoding="utf-8")
    
    print(f"\n{'=' * 60}")
    print(f"Dataset saved to: {output_path}")
    print(f"Total records: {len(df)}")
    print(f"\nClass Distribution:")
    print(df["Language"].value_counts().to_string())
    print(f"\nSample records:")
    print(df.head(10).to_string())
    print(f"{'=' * 60}")
    
    return df


if __name__ == "__main__":
    create_dataset()
