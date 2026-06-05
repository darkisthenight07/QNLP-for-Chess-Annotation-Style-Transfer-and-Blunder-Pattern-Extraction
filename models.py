"""
models.py
SQLAlchemy ORM models for the QNLP Chess project.

Tables:
  games            — raw PGN game metadata
  moves            — individual moves per game
  annotations      — extracted { } comments linked to moves
  blunder_labels   — human/model blunder category labels
  corpus_sentences — cleaned sentences ready for QNLP encoding
"""

from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    Float,
    ForeignKey,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

# ------------------------------------------------------------------
# Games
# ------------------------------------------------------------------

class Game(Base):
    __tablename__ = "games"

    id = Column(Integer, primary_key=True)

    white_player = Column(String(100))
    black_player = Column(String(100))
    result = Column(String(10))

    moves = relationship(
        "Move",
        back_populates="game",
        cascade="all, delete-orphan"
    )


# ------------------------------------------------------------------
# Moves
# ------------------------------------------------------------------

class Move(Base):
    __tablename__ = "moves"

    id = Column(Integer, primary_key=True)

    game_id = Column(
        Integer,
        ForeignKey("games.id"),
        nullable=False
    )

    move_number = Column(Integer)
    san = Column(String(20))

    game = relationship(
        "Game",
        back_populates="moves"
    )

    annotation = relationship(
        "Annotation",
        back_populates="move",
        uselist=False,
        cascade="all, delete-orphan"
    )


# ------------------------------------------------------------------
# Chess Comments
# ------------------------------------------------------------------

class Annotation(Base):
    __tablename__ = "annotations"

    id = Column(Integer, primary_key=True)

    move_id = Column(
        Integer,
        ForeignKey("moves.id"),
        nullable=False
    )

    text = Column(Text, nullable=False)

    move = relationship(
        "Move",
        back_populates="annotation"
    )

    sentences = relationship(
        "CorpusSentence",
        back_populates="annotation",
        cascade="all, delete-orphan"
    )

    labels = relationship(
        "BlunderLabel",
        back_populates="annotation",
        cascade="all, delete-orphan"
    )


# ------------------------------------------------------------------
# Sentences for QNLP
# ------------------------------------------------------------------

class CorpusSentence(Base):
    __tablename__ = "corpus_sentences"

    id = Column(Integer, primary_key=True)

    annotation_id = Column(
        Integer,
        ForeignKey("annotations.id"),
        nullable=False
    )

    sentence = Column(Text, nullable=False)

    annotation = relationship(
        "Annotation",
        back_populates="sentences"
    )

# ------------------------------------------------------------------
# Blunder Categories
# ------------------------------------------------------------------

class BlunderLabel(Base):
    __tablename__ = "blunder_labels"

    id = Column(Integer, primary_key=True)

    annotation_id = Column(
        Integer,
        ForeignKey("annotations.id"),
        nullable=False
    )

    category = Column(String(50))
    confidence = Column(Float)

    annotation = relationship(
        "Annotation",
        back_populates="labels"
    )