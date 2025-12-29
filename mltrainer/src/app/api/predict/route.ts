import { NextResponse } from "next/server";
import { exec } from "child_process";
import path from "path";

// Store the server process reference if we want to manage it (simple version)
// Note: In a real app, we'd use a proper process manager or separate service
let serverProcess: any = null;

export async function POST(req: Request) {
  try {
    const body = await req.json();
    const { action, modelPath, port = 8000 } = body;

    if (action === "start") {
      if (!modelPath) {
        return NextResponse.json(
          { error: "Model path required" },
          { status: 400 },
        );
      }

      // Kill existing server if any (very crude)
      if (serverProcess) {
        try {
          process.kill(-serverProcess.pid); // Kill process group
        } catch (e) {
          // Ignore
        }
      }

      const rootDir = path.resolve(process.cwd(), "..");
      const cmd = `cd ${rootDir} && python3 -m synthai.cli serve "${modelPath}" --port ${port}`;

      console.log(`Starting server: ${cmd}`);

      // Spawn process detached so it keeps running
      const { spawn } = require("child_process");
      serverProcess = spawn("sh", ["-c", cmd], {
        detached: true,
        stdio: "ignore",
      });

      serverProcess.unref();

      // Wait a bit to ensure it starts
      await new Promise((resolve) => setTimeout(resolve, 2000));

      return NextResponse.json({ message: "Server started", port });
    } else if (action === "predict") {
      // Proxy request to the running python server
      const { data } = body;

      try {
        const response = await fetch(`http://localhost:${port}/predict`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ data }),
        });

        if (!response.ok) {
          const errorText = await response.text();
          throw new Error(`Prediction failed: ${errorText}`);
        }

        const result = await response.json();
        return NextResponse.json(result);
      } catch (error: any) {
        return NextResponse.json({ error: error.message }, { status: 500 });
      }
    }

    return NextResponse.json({ error: "Invalid action" }, { status: 400 });
  } catch (error: any) {
    return NextResponse.json(
      { error: error.message || "Internal server error" },
      { status: 500 },
    );
  }
}
