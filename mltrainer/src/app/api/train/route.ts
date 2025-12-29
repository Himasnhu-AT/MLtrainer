import { NextResponse } from "next/server";
import { exec } from "child_process";
import path from "path";

export async function POST(req: Request) {
  try {
    const body = await req.json();
    const { data, schema, modelType, output, tune, iterations } = body;

    if (!data || !schema) {
      return NextResponse.json(
        { error: "Missing required fields: data, schema" },
        { status: 400 },
      );
    }

    // Construct command
    // We assume we are running from mltrainer directory, so we need to go up to root
    // Also assuming python is available or venv is activated
    const rootDir = path.resolve(process.cwd(), "..");
    const pythonCmd = "python3 -m synthai.cli train";

    let cmd = `cd ${rootDir} && ${pythonCmd} --data "${data}" --schema "${schema}"`;

    if (modelType) cmd += ` --model-type "${modelType}"`;
    if (output) cmd += ` --output "${output}"`;
    if (tune) cmd += ` --tune`;
    if (iterations) cmd += ` --iterations ${iterations}`;

    console.log(`Executing command: ${cmd}`);

    return new Promise((resolve) => {
      exec(cmd, (error, stdout, stderr) => {
        if (error) {
          console.error(`Error: ${error.message}`);
          resolve(
            NextResponse.json(
              { error: error.message, stderr },
              { status: 500 },
            ),
          );
          return;
        }

        console.log(`Stdout: ${stdout}`);
        resolve(
          NextResponse.json({
            message: "Training completed successfully",
            stdout,
          }),
        );
      });
    });
  } catch (error) {
    return NextResponse.json(
      { error: "Internal server error" },
      { status: 500 },
    );
  }
}
