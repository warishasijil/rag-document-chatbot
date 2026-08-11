from pathlib import Path

import pandas as pd
from langchain_core.documents import Document

from rag_chatbot.ingestion.base_loader import BaseDocumentLoader


class ExcelDocumentLoader(BaseDocumentLoader):
    """Loader for Excel (.xlsx) workbooks."""

    def load(self, file_path: Path) -> list[Document]:
        workbook = pd.ExcelFile(file_path)

        documents: list[Document] = []

        for sheet_name in workbook.sheet_names:
            dataframe = pd.read_excel(
                file_path,
                sheet_name=sheet_name,
            )

            # Remove completely empty rows
            dataframe = dataframe.dropna(how="all")

            for row_number, (_, row) in enumerate(
                dataframe.iterrows(),
                start=2,
            ):
                row_content = []

                for column_name, value in row.items():
                    if pd.isna(value):
                        continue

                    row_content.append(
                        f"{column_name}: {value}"
                    )

                if not row_content:
                    continue

                text = "\n".join(row_content)

                document = Document(
                    page_content=text,
                    metadata={
                        "source": str(file_path),
                        "file_name": file_path.name,
                        "file_type": file_path.suffix.lower(),
                        "sheet": sheet_name,
                        "row": row_number,
                    },
                )

                documents.append(document)

        return documents