import asyncio
import websockets

class WebsocketServer:
    _websocket_clients = set()

    def __init__(self, port=61002):
        start_server = websockets.serve(self._websocket_open, '0.0.0.0', port)
        loop = asyncio.get_event_loop()
        loop.run_until_complete(start_server)
        print('websocket server started')
        loop.run_forever()

    async def _websocket_open(self, websocket, path):
        try:
            async for message in websocket:
                if message == 'actor':
                    self._websocket_clients.add(websocket)
                    self.print_websocket_clients()
                    await websocket.send('reset')
        except websockets.ConnectionClosed:
            try:
                self._websocket_clients.remove(websocket)
            except:
                pass
            else:
                self.print_websocket_clients()

    def print_websocket_clients(self):
        log_str = f'{len(self._websocket_clients)} active actors'
        for i, client in enumerate(self._websocket_clients):
            log_str += (f'\n[{i+1}]. {client.remote_address[0]} : {client.remote_address[1]}')

        print(log_str)

    async def send_to_all(self, message):
        tasks = []
        try:
            for client in self._websocket_clients:
                tasks.append(client.send(message))
            await asyncio.gather(*tasks)
        except websockets.ConnectionClosed:
            pass

WebsocketServer()