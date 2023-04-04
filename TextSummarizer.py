import spacy
import pytextrank


from transformers import PegasusForConditionalGeneration, PegasusTokenizer, pipeline
from bs4 import BeautifulSoup
import requests


class TextSummarizer:

    def __init__(self):
        self.paragraphArray = []

    def summarizeSentence(self,ARTICLE):

        summarizer = pipeline("summarization")

        # ARTICLE = r"Currently, computers are made with silicon transistors. These transistors are getting smaller and more powerful with each passing year. However, there is a physical limit to this technology. In these types of circuits, if the conductors are too close to each other, electrons can bounce between them. Moreover, if a transistor is too small, electrons can go through the gate of the transistor. This phenomenon is known as quantum tunneling and can ruin the entire circuit. It’s clear that the uncertain behavior of quantum particles is the basis of the physical limits of silicon circuits. Scientists invented a new computer technology with this uncertain behavior of quantum particles, known as quantum computing. Even though quantum computing isn’t an absolute replacement for silicon computers, in specific cases, it can provide unbelievable processing power. Bits are used to store data in a computer. When the transistor is active, it’s a 1 and when inactive it’s a 0. The formula 2^bits represents the number of combinations bits can create. A quantum particle is used as a bit in quantum computers. It’s called a qubit. It could be an electron, photon, or any particle, but outer electrons in phosphorus atoms are commonly used. Here, it’s a 1 when the spin is up and 0 when it’s down. The spin can be controlled with an electromagnetic field. So far, it seems like quantum and silicon computers are very similar, but quantum particles are mysterious. The spin of electrons can be up, down, and when we’re not observing, they can be up and down at the same time. This is known as quantum superposition. Hence, quantum computers can provide incredible processing power. In a classical computer, if we have 2 bits, it can create 4 combinations, yet use only one at a time. Also, 4 combinations can be created by 2 qubits. Due to superposition, it can use them all at the same time. With 20 and 60 qubits, it can maintain one million combinations and any number of combinations equal to all the particles in the universe respectively. Because of this, it’s clear that a quantum computer is not a substitution for a classical computer. They are particularly designed for parallel processing. Classical computers use bits (0 or 1) to process information. But if we use quantum particles as data, something interesting happens. By using superposition, they can read both as a 0 or a 1 at the same time. This makes the amount of data that can be represented exponentially greater and allows quantum computers to process far more data than classical computers will ever be able to do. If a quantum computer had 100 qubits, it would be more powerful for some applications than all of the supercomputers on earth combined. Three hundred qubits could hold more numbers simultaneously than there are atoms in the universe. So, think about what a billion qubits would be able to do. Entanglement is another phenomenon where two particles can be linked so that one particle always gives the same outcome as the other even if they are separated on opposite sides of the earth or even the universe. They would show the same result as each other every single time. It’s still being debated, but entangled particles could make communication instant, regardless of the distance between the particles. It would be great for security as well since it potentially doesn’t use any physical infrastructure to transfer this information. This means that in the future, it may be impossible for communication to be intercepted or hacked without the knowledge of the information owner. "
        max_chunk = 400
        ARTICLE = ARTICLE.replace('.', '.<eos>')
        ARTICLE = ARTICLE.replace('?', '?<eos>')
        ARTICLE = ARTICLE.replace('!', '!<eos>')

        sentences = ARTICLE.split('<eos>')
        current_chunk = 0
        chunks = []
        for sentence in sentences:
            if len(chunks) == current_chunk + 1:
                if len(chunks[current_chunk]) + len(sentence.split(' ')) <= max_chunk:
                    chunks[current_chunk].extend(sentence.split(' '))
                else:
                    current_chunk += 1
                    chunks.append(sentence.split(' '))
            else:
                print(current_chunk)
                chunks.append(sentence.split(' '))

        for chunk_id in range(len(chunks)):
            chunks[chunk_id] = ' '.join(chunks[chunk_id])


        print(len(chunks))


        res = summarizer(chunks, max_length=100, min_length=30, do_sample=False)

        summ = ' '.join([summ['summary_text'] for summ in res])

        self.paragraphArray.append(summ)

    def returntheSummarizePara(self):
        tempStr = ""
        for sent in self.paragraphArray:
            tempStr+= sent
        return tempStr


# ------------------------------------------------------------------------------------------



# nlp = spacy.load("en_core_web_lg")
# nlp.add_pipe("textrank")
#
# texts = ["Quantum computing is an area of computer science that uses the principles of quantum theory. Quantum theory explains the behavior of energy and material on the atomic and subatomic levels.","Quantum computing uses subatomic particles, such as electrons or photons. Quantum bits, or qubits, allow these particles to exist in more than one state (i.e., 1 and 0) at the same time.","Classical computers today employ a stream of electrical impulses (1 and 0) in a binary manner to encode information in bits. This restricts their processing ability, compared to quantum computing.","Quantum computing has the capability to sift through huge numbers of possibilities and extract potential solutions to complex problems and challenges. Where classical computers store information as bits with either 0s or 1s, quantum computers use qubits. Qubits carry information in a quantum state that engages 0 and 1 in a multidimensional way"]
# # doc = nlp(text)
# #
# # for sent in doc._.textrank.summary(limit_sentences =2):
# #     print(sent)
#
# print("Model Name")
# model_name = "google/pegasus-xsum"
#
# print("pegasus_tokenizer")
# pegasus_tokenizer = PegasusTokenizer.from_pretrained(model_name)
#
# print("Pegasus model")
# pegasus_model = PegasusForConditionalGeneration.from_pretrained(model_name,max_length=5120)
#
#
# for text in texts:
#     print("Tokens")
#     tokens = pegasus_tokenizer(text,truncation=True,padding="longest",return_tensors="pt")
#
#     print("Encoded Sum")
#     encodedSummary = pegasus_model.generate(**tokens,max_length=1024)
#
#     print("Decoded")
#     decodedSummary = pegasus_tokenizer.decode(encodedSummary[0],skip_special_tokens=True)
#
#     print(decodedSummary)
