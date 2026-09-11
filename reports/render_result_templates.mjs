import fs from "node:fs/promises";
import path from "node:path";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const root = process.cwd();
const sourceDir = path.join(root, "data", "result-data");
const outputDir = path.join(root, "reports", "template_previews");
await fs.mkdir(outputDir, { recursive: true });

for (const name of ["result1.xlsx", "result2.xlsx", "result3.xlsx", "result4-2.xlsx", "result4-3.xlsx"]) {
  const input = await FileBlob.load(path.join(sourceDir, name));
  const workbook = await SpreadsheetFile.importXlsx(input);
  const firstSheet = workbook.worksheets.getItemAt(0);
  const preview = await workbook.render({
    sheetName: firstSheet.name,
    range: firstSheet.getUsedRange().address,
    scale: 1,
    format: "png",
  });
  await fs.writeFile(
    path.join(outputDir, name.replace(".xlsx", ".png")),
    new Uint8Array(await preview.arrayBuffer()),
  );
}
