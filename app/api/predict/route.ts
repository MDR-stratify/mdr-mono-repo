export async function POST(request: Request) {
  try {
    const body = await request.json();
    
    // Forward the request to the Python API
    const response = await fetch('https://stratify-api-production.up.railway.app/predict', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body),
    });

    if (!response.ok) {
      throw new Error('Failed to get prediction from API');
    }

    const data = await response.json();
    return Response.json(data);
  } catch (error) {
    console.error('Prediction error:', error);
    return Response.json(
      { error: 'Failed to get prediction' },
      { status: 500 }
    );
  }
}