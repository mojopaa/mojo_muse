import click


@click.group()
def main():
    pass


@main.command()
def init():
    print("init")


if __name__ == "__main__":
    main()
