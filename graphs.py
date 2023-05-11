import matplotlib.pyplot as plt


def graph1(df):
    # Crear una lista de colores para los puntos
    colors = ['blue' if x == -1 else 'red' for x in df['status']]

    # Crear el gráfico de dispersión
    plt.scatter(df['ratio_intMedia'], df['ratio_intHyperlinks'], c=colors)

    # Agregar etiquetas de los ejes x e y
    plt.xlabel('ratio_intMedia')
    plt.ylabel('ratio_intHyperlinks')

    # Mostrar el gráfico
    plt.show(block=True)


def graph2(df):
    colors = ['blue' if x == -1 else 'red' for x in df['status']]
    plt.scatter(df['page_rank'], df['ratio_extMedia'], c=colors)
    plt.xlabel('page_rank')
    plt.ylabel('ratio_extMedia')
    plt.show(block=True)


def graph3(df):
    colors = ['blue' if x == -1 else 'red' for x in df['status']]
    plt.scatter(df['ratio_intHyperlinks'], df['page_rank'], c=colors)
    plt.xlabel('ratio_intHyperlinks')
    plt.ylabel('page_rank')
    plt.show(block=True)


def graph4(df):
    colors = ['blue' if x == -1 else 'red' for x in df['status']]

    plt.scatter(df['ratio_intMedia'], df['ratio_extMedia'], c=colors)
    plt.xlabel('ratio_intMedia')
    plt.ylabel('ratio_extMedia')
    plt.show(block=True)
