import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';

export async function GET() {
    try {
        // Default models directory relative to root
        const rootDir = path.resolve(process.cwd(), '..');
        const modelsDir = path.join(rootDir, 'models');

        if (!fs.existsSync(modelsDir)) {
            return NextResponse.json({ models: [] });
        }

        const files = fs.readdirSync(modelsDir);

        // Filter for .pkl files
        const models = files
            .filter(file => file.endsWith('.pkl') && !file.startsWith('preprocessor_'))
            .map(file => {
                const stats = fs.statSync(path.join(modelsDir, file));
                return {
                    name: file,
                    path: path.join('models', file), // Relative path for CLI
                    fullPath: path.join(modelsDir, file),
                    size: stats.size,
                    created: stats.birthtime
                };
            });

        return NextResponse.json({ models });

    } catch (error) {
        console.error('Error listing models:', error);
        return NextResponse.json(
            { error: 'Internal server error' },
            { status: 500 }
        );
    }
}
