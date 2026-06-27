import os
import xml.etree.ElementTree as ET
import spacy
from collections import Counter
from spacy.util import filter_spans
from spacy.training import offsets_to_biluo_tags


nlp = spacy.blank("pt")
nlp.add_pipe("sentencizer")

GRUPOS_PARA_STY = {
    'ANAT': [  # Anatomy
        'BodyLocationorRegion', 'BodyPart,Organ,orOrganComponent',
        'BodySpaceorJunction', 'BodySubstance', 'BodySystem', 'Cell', 'Tissue',
    ],
    'CHEM': [  # Chemicals & Drugs
        'PharmacologicSubstance', 'OrganicChemical', 'AminoAcid,Peptide,orProtein',
        'Hormone', 'Antibiotic', 'BiologicallyActiveSubstance', 'InorganicChemical',
        'ClinicalDrug', 'HazardousorPoisonousSubstance',
        'NucleicAcid,Nucleoside,orNucleotide', 'ImmunologicFactor',
        'BiomedicalorDentalMaterial', 'Vitamin', 'Enzyme', 'Element,Ion,orIsotope',
    ],
    'DISO': [  # Disorders
        'Finding', 'SignorSymptom', 'DiseaseorSyndrome', 'InjuryorPoisoning',
        'PathologicFunction', 'NeoplasticProcess', 'MentalorBehavioralDysfunction',
        'AnatomicalAbnormality', 'AcquiredAbnormality', 'CongenitalAbnormality',
        'CellorMolecularDysfunction',
    ],
    'PROC': [  # Procedures
        'TherapeuticorPreventiveProcedure', 'HealthCareActivity',
        'DiagnosticProcedure', 'LaboratoryProcedure',
        'EducationalActivity', 'ResearchActivity',  # PROC no SemGroups.txt oficial
    ],
    'DEVI': [  # Devices
        'MedicalDevice', 'DrugDeliveryDevice',
    ],
    'PHYS': [  # Physiology
        'ClinicalAttribute', 'PhysiologicFunction', 'OrganismFunction',
        'OrganismAttribute', 'OrganorTissueFunction', 'MentalProcess',
        'MolecularFunction',
    ],
    'PHEN': [  # Phenomena
        'LaboratoryorTestResult', 'PhenomenonorProcess', 'NaturalPhenomenonorProcess',
    ],
    'LIVB': [  # Living Beings
        'PatientorDisabledGroup', 'ProfessionalorOccupationalGroup', 'FamilyGroup',
        'Bacterium', 'PopulationGroup', 'AgeGroup', 'Virus', 'Fungus', 'Fish',
        'Group', 'Plant',
    ],
    'CONC': [  # Concepts & Ideas
        'QuantitativeConcept', 'TemporalConcept', 'QualitativeConcept',
        'SpatialConcept', 'IntellectualProduct', 'FunctionalConcept',
        'Classification', 'IdeaorConcept', 'RegulationorLaw',
    ],
    'ACTI': [  # Activities & Behaviors
        'DailyorRecreationalActivity', 'IndividualBehavior', 'OccupationalActivity',
        'Event', 'MachineActivity', 'Activity', 'SocialBehavior', 'Behavior',
    ],
    'OBJC': [  # Objects
        'Food', 'ManufacturedObject', 'Substance', 'PhysicalObject',
    ],
    'ORGA': [  # Organizations
        'HealthCareRelatedOrganization', 'Organization',
    ],
    'OCCU': [  # Occupations
        'BiomedicalOccupationorDiscipline',
    ],
    'GENE': [  # Genes & Molecular Sequences
        'AminoAcidSequence',
    ],
}

#Recebe um STY e retorna seu Grupo
STY_PARA_GRUPO = {}
for _grupo, _stys in GRUPOS_PARA_STY.items():
    for _sty in _stys:
        STY_PARA_GRUPO[_sty] = _grupo

#Prioridade para quando existem rótulos de grupos diferentes na mesma anotação
PRIORIDADE_GRUPOS = [
    'DISO', 'CHEM', 'PROC', 'DEVI', 'ANAT', 'PHYS', 'PHEN', 'LIVB',
    'CONC', 'ACTI', 'OBJC', 'ORGA', 'OCCU', 'GENE', 'GEOG',
]
RANK_GRUPO = {}
indice = 0
for g in PRIORIDADE_GRUPOS:
    RANK_GRUPO[g] = indice
    indice += 1


def mapear_para_grupo(tag_original):
    if not tag_original:
        return None

    grupos_encontrados = set()
    
    #Separa os STYs divididos por "|" e busca o grupo de cada um
    for parte in tag_original.split('|'):
        sty = parte.strip().replace(' ', '')
        grupo = STY_PARA_GRUPO.get(sty)
        if grupo is not None:
            grupos_encontrados.add(grupo)

    if not grupos_encontrados:
        return None

    #Qual grupo encontrado tem a maior prioridade
    melhor_grupo = None
    melhor_posicao = 999999  
    
    for grupo in grupos_encontrados:
        #Pega a posição do grupo na lista de prioridades
        if grupo in RANK_GRUPO:
            posicao = RANK_GRUPO[grupo]
        else:
            #Se não existir na lista, tem uma posição ruim
            posicao = len(PRIORIDADE_GRUPOS)
            
        #Se achamos uma posição menor (melhor prioridade), guardamos ela
        if posicao < melhor_posicao:
            melhor_posicao = posicao
            melhor_grupo = grupo
            
    return melhor_grupo


def ler_texto_raw_xml(caminho_arquivo):
    #Isso é necessário para não perder os caracteres invisíveis de quebra de linha do Windows (\\r\\n).
    with open(caminho_arquivo, 'rb') as arquivo:
        conteudo = arquivo.read()
        
    marcador_inicio = b'<TEXT>'
    marcador_fim = b'</TEXT>'
    
    idx_inicio = conteudo.find(marcador_inicio)
    idx_fim = conteudo.find(marcador_fim)
    
    if idx_inicio == -1 or idx_fim == -1:
        return None
        
    #Pega só o texto que está entre <TEXT> e </TEXT> e converte de volta para texto
    texto_bytes = conteudo[idx_inicio + len(marcador_inicio):idx_fim]
    return texto_bytes.decode('utf-8')


def converter_offset_crlf_para_lf(texto_crlf, offset_crlf):
    #Subtrai da posição original a quantidade de caracteres '\\r' que apareceram antes dela.
    quantidade_cr_antes = texto_crlf[:offset_crlf].count('\r')
    posicao_corrigida = offset_crlf - quantidade_cr_antes
    return posicao_corrigida


def extrair_anotacoes_do_xml(raiz_xml, texto_lf, texto_crlf, contador_grupos):
    #Percorre todas as tags <annotation> do XML, corrige as posições e garante que não comecem nem terminem com espaços em branco
    anotacoes_limpas = []

    for ann in raiz_xml.findall('.//annotation'):
        tag_original = ann.get('tag')

        #Descobre qual é o grupo correto para essa tag
        etiqueta_grupo = mapear_para_grupo(tag_original)
        if etiqueta_grupo is None:
            continue

        if contador_grupos is not None:
            contador_grupos[etiqueta_grupo] += 1

        #Pega as posições anotadas no XML (feitas no texto com \r\n)
        inicio_crlf = int(ann.get('start'))
        fim_crlf = int(ann.get('end'))

        #Converte essas posições para o texto do Python (que só tem \n)
        inicio_lf = converter_offset_crlf_para_lf(texto_crlf, inicio_crlf)
        fim_lf = converter_offset_crlf_para_lf(texto_crlf, fim_crlf)

        texto_anotado = texto_lf[inicio_lf:fim_lf]

        #Cortar eventuais espaços vazios que ficaram nas pontas.
        texto_sem_espaco_esquerdo = texto_anotado.lstrip()
        
        espacos_removidos_esq = len(texto_anotado) - len(texto_sem_espaco_esquerdo)
        inicio_final = inicio_lf + espacos_removidos_esq

        # rstrip() remove os espaços da direita
        texto_final_limpo = texto_sem_espaco_esquerdo.rstrip()
        # O 'fim' será o início corrigido + o tamanho da palavra limpa
        fim_final = inicio_final + len(texto_final_limpo)

        # Só consideramos a anotação se sobrou algum texto
        if inicio_final < fim_final:
            anotacoes_limpas.append((inicio_final, fim_final, etiqueta_grupo))

    return anotacoes_limpas


def converter_biluo_para_bio(tag_biluo):
    #Tradução do Biluo para BIO

    if tag_biluo in ("O", "-"):
        return "O"
    
    #Se já for B- (começo) ou I- (dentro), mantemos igual
    if tag_biluo.startswith("B-") or tag_biluo.startswith("I-"):
        return tag_biluo
        
    #No BIO, 'L' vira um 'I'
    if tag_biluo.startswith("L-"):
        return "I-" + tag_biluo[2:]
        
    #No BIO, 'U' vira um 'B'
    if tag_biluo.startswith("U-"):
        return "B-" + tag_biluo[2:]
        
    return "O"


def processar_texto_com_spacy(texto_lf, anotacoes_corrigidas):
    #Analisa o texto com o spaCy (quebra em sentenças e tokens)
    documento = nlp(texto_lf)

    #Transforma posições (início, fim) em "Spans" que o spaCy entende
    lista_spans = []
    for inicio, fim, etiqueta in anotacoes_corrigidas:
        #Ignora anotações que cortam palavras no meio
        span = documento.char_span(inicio, fim, label=etiqueta, alignment_mode="contract")
        if span is not None:
            lista_spans.append(span)

    #Se houver anotações encavaladas (sobrepostas), escolhe a melhor e descarta a outra
    spans_sem_conflito = filter_spans(lista_spans)

    #Prepara a lista final de entidades como o spaCy espera (inicio, fim, rotulo)
    entidades_formatadas = []
    for s in spans_sem_conflito:
        entidades_formatadas.append((s.start_char, s.end_char, s.label_))

    #Atribui uma tag para CADA TOKEN do texto (no formato BILUO)
    tags_biluo = offsets_to_biluo_tags(documento, entidades_formatadas)

    #Monta a lista final, separando as sentenças
    sentencas_processadas = []
    
    for sentenca_doc in documento.sents:
        sentenca_atual = []

        for token in sentenca_doc:
            #Ignora espaços vazios
            if token.is_space:
                continue

            indice_token = token.i
            tag_do_token = tags_biluo[indice_token]

            #Traduz BILUO para BIO
            tag_bio = converter_biluo_para_bio(tag_do_token)
            sentenca_atual.append((token.text, tag_bio))

        if len(sentenca_atual) > 0:
            sentencas_processadas.append(sentenca_atual)

    return sentencas_processadas


def processar_arquivo(caminho_arquivo, contador_grupos=None):
    try:
        arvore = ET.parse(caminho_arquivo)
        raiz = arvore.getroot()
        elemento_texto = raiz.find('TEXT')

        #Se não houver texto, retorna vazio
        if elemento_texto is None or elemento_texto.text is None:
            return []

        #O texto lido pelo ET.parse (com \n normalizado)
        texto_lf = elemento_texto.text

        #Lê o texto cru (com \r\n original) para poder arrumar o início/fim
        texto_crlf = ler_texto_raw_xml(caminho_arquivo)
        if texto_crlf is None:
            texto_crlf = texto_lf
            
        #Pega as anotações e ajusta as posições e espaços em branco
        anotacoes_corrigidas = extrair_anotacoes_do_xml(raiz, texto_lf, texto_crlf, contador_grupos)
        
        #Pede para o spaCy tokenizar e cruzar com as anotações
        sentencas_anotadas = processar_texto_com_spacy(texto_lf, anotacoes_corrigidas)

        return sentencas_anotadas

    except Exception as erro:
        print(f"Erro ao processar o arquivo {caminho_arquivo}: {erro}")
        return []


def salvar_bio(sentencas, caminho_arquivo):
    with open(caminho_arquivo, 'w', encoding='utf-8') as arquivo_saida:
        for sentenca in sentencas:
            if not sentenca:
                continue
            
            for palavra, rotulo in sentenca:
                arquivo_saida.write(f"{palavra} {rotulo}\n")
                
            #Adiciona uma linha em branco ao fim de cada sentença
            arquivo_saida.write("\n")


def main():
    diretorios = ['train', 'test', 'val']

    #Caminhos para as pastas
    diretorio_script = os.path.dirname(os.path.abspath(__file__))
    diretorio_base = os.path.join(diretorio_script, '..')

    diretorio_bruto = os.path.join(diretorio_base, 'SemClinBr-xml-public-v1')
    diretorio_processado = os.path.join(diretorio_base, 'dataset', 'grupos_sgr')

    #Cria a pasta de saída se não existir
    os.makedirs(diretorio_processado, exist_ok=True)

    contador_global = Counter()

    #Passa por treino, validação e teste
    for nome_pasta in diretorios:
        dir_entrada = os.path.join(diretorio_bruto, nome_pasta)
        
        if not os.path.exists(dir_entrada):
            print(f"Aviso: Diretório {dir_entrada} não encontrado. Pulando...")
            continue

        arquivo_saida = os.path.join(diretorio_processado, f"{nome_pasta}.bio")
        todas_as_sentencas_da_pasta = []

        #Lista apenas os arquivos terminados em .xml e ordena alfabeticamente
        arquivos = []
        for arquivo in os.listdir(dir_entrada):
            if arquivo.endswith('.xml'):
                arquivos.append(arquivo)
                
        arquivos.sort()

        print(f"Processando {len(arquivos)} arquivos do diretório '{nome_pasta}'...")

        for nome_arquivo in arquivos:
            caminho_completo = os.path.join(dir_entrada, nome_arquivo)
            
            sentencas_deste_arquivo = processar_arquivo(caminho_completo, contador_global)
            
            #Adiciona as sentenças processadas na lista da pasta
            todas_as_sentencas_da_pasta.extend(sentencas_deste_arquivo)

        print(f"Total de sentenças extraídas em '{nome_pasta}': {len(todas_as_sentencas_da_pasta)}")

        #Salva o arquivo BIO correspondente
        salvar_bio(todas_as_sentencas_da_pasta, arquivo_saida)
        print(f"Arquivo salvo com sucesso em: {arquivo_saida}\n")

    #Imprime uma estatística final dos rótulos mapeados
    print("\nDistribuição final dos grupos semânticos SGR:")
    for grupo, quantidade in contador_global.most_common():
        print(f"  {grupo}\t{quantidade}")

if __name__ == "__main__":
    main()
